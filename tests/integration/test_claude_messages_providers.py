"""Claude API and Amazon Bedrock providers: request shaping, wire choice, and recovery typing.

The official Anthropic SDK runs for real against a mocked transport, so these
tests cover the bytes MOTO sends and the event streams it reads back.
"""
import json
import os
import tempfile
from typing import Callable, List
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from fastapi import HTTPException
import httpx

from backend.api.routes import cloud_access as cloud_access_route
from backend.shared.config import system_config
from backend.shared.model_error_utils import is_provider_context_length_error, is_transient_model_call_error
from backend.shared.models import ModelConfig
from backend.shared.subscription_providers.amazon_bedrock import AMAZON_BEDROCK_SETTINGS, AmazonBedrockProfile
from backend.shared.subscription_providers.api_key_client import SubscriptionApiKeyClient
from backend.shared.subscription_providers.claude_api import CLAUDE_API_SETTINGS, ClaudeApiProfile
from backend.shared.subscription_providers.claude_messages import (
    ClaudeModelTraits,
    claude_family_id,
    convert_messages,
    known_claude_traits,
    resolve_claude_effort,
)
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionParameterRejectedError,
    SubscriptionRequestError,
)
from backend.shared.subscription_providers.registry import get_api_key_subscription_client
from tests.integration.claude_messages_support import (
    bridged_messages_client,
    messages_error,
    messages_response,
    messages_stream_error,
)

_CLIENT_MODULE = "backend.shared.subscription_providers.api_key_client"
CLAUDE_KEY = "sk-claude-test-0123456789abcdef"
BEDROCK_KEY = "bedrock-api-key-test-0123456789abcdef"
MANTLE = "https://bedrock-mantle.{region}.api.aws"
USER = [{"role": "user", "content": "hi"}]
LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a value up",
        "parameters": {"type": "object", "properties": {"q": {"type": "integer"}}},
    },
}


def chat_sse(text: str = "OK") -> bytes:
    chunks = [
        {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode("utf-8")


class ClaudeEffortAndModelIdentityTests(TestCase):
    def test_family_id_strips_platform_prefixes_and_version_suffixes(self) -> None:
        self.assertEqual(claude_family_id("claude-opus-5-5"), "claude-opus-5-5")
        self.assertEqual(claude_family_id("global.anthropic.claude-opus-5-5"), "claude-opus-5-5")
        self.assertEqual(claude_family_id("anthropic.claude-haiku-4-5-20251001-v1:0"), "claude-haiku-4-5")
        self.assertEqual(claude_family_id("openai.gpt-oss-120b"), "")

    def test_effort_is_never_raised_above_the_request(self) -> None:
        opus = known_claude_traits("claude-opus-5-5")
        self.assertIsNone(resolve_claude_effort("auto", opus))
        self.assertIsNone(resolve_claude_effort("", opus))
        self.assertEqual(resolve_claude_effort("max", opus), "max")
        self.assertEqual(resolve_claude_effort("xhigh", opus), "xhigh")
        self.assertEqual(resolve_claude_effort("minimal", opus), "low")
        # Thinking cannot be turned off on several current models.
        self.assertEqual(resolve_claude_effort("none", opus), "low")
        # Opus 4.6 has no xhigh; the request falls to the next lower level, not to max.
        self.assertEqual(resolve_claude_effort("xhigh", known_claude_traits("claude-opus-4-6")), "high")
        # Haiku 4.5 takes no effort setting at all.
        self.assertIsNone(resolve_claude_effort("max", known_claude_traits("claude-haiku-4-5")))
        # An unknown model gets the request as given.
        self.assertEqual(resolve_claude_effort("high", None), "high")
        self.assertIsNone(resolve_claude_effort("bogus", None))

    def test_message_conversion_keeps_order_and_never_sends_empty_text(self) -> None:
        system, turns = convert_messages([
            {"role": "system", "content": "rules"},
            {"role": "developer", "content": "more rules"},
            {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "draft"},
            {"role": "system", "content": "later instruction"},
            {"role": "user", "content": "retry"},
        ])
        self.assertEqual(system, "rules\n\nmore rules")
        self.assertEqual([turn["role"] for turn in turns], ["user", "assistant", "user"])
        self.assertEqual(turns[0]["content"], [{"type": "text", "text": "part one\npart two"}])
        self.assertEqual(turns[1]["content"], [{"type": "text", "text": "draft"}])
        self.assertEqual(
            [block["text"] for block in turns[2]["content"]],
            ["later instruction", "retry"],
        )


class ClaudeHarness(IsolatedAsyncioTestCase):
    profile_factory: Callable = staticmethod(lambda: ClaudeApiProfile(CLAUDE_API_SETTINGS))
    key = CLAUDE_KEY

    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(system_config, "data_dir", self._data_dir.name),
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch(f"{_CLIENT_MODULE}.load_subscription_api_key", return_value=None),
            # Restored on teardown; the developer's own AWS region must not pick the endpoint.
            mock.patch.dict("os.environ"),
        ]
        for patcher in self._patches:
            patcher.start()
        for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "MOTO_AMAZON_BEDROCK_REGION", "MOTO_AMAZON_BEDROCK_BASE_URL",
                     "MOTO_CLAUDE_API_BASE_URL"):
            os.environ.pop(name, None)
        self.profile = self.profile_factory()
        self.client = SubscriptionApiKeyClient(self.profile)
        self.client._api_key = self.key
        self.client._persisted_region_checked = True
        self.requests: List[httpx.Request] = []
        self.wait = mock.AsyncMock(return_value=True)
        wait_patch = mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=self.wait)
        wait_patch.start()
        self._patches.append(wait_patch)

    def tearDown(self) -> None:
        mock.patch.stopall()
        self._data_dir.cleanup()

    def use(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        def recording(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        self.client.client = httpx.AsyncClient(transport=httpx.MockTransport(recording))
        self.client.messages_http_client = bridged_messages_client(recording)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)

    async def complete(self, **kwargs):
        kwargs.setdefault("model", "claude-opus-5-5")
        kwargs.setdefault("messages", USER)
        kwargs.setdefault("max_tokens", 4000)
        return await self.client.generate_completion(**kwargs)


class ClaudeApiRequestTests(ClaudeHarness):
    async def test_request_uses_the_messages_api_and_identifies_moto(self) -> None:
        self.use(lambda request: messages_response("Hello"))
        result = await self.complete(
            messages=[{"role": "system", "content": "be exact"}, *USER],
            reasoning_effort="xhigh",
            temperature=0.4,
        )
        self.assertEqual(result["choices"][0]["message"], {"role": "assistant", "content": "Hello"})
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"], {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        self.assertEqual(result["_moto_wire_api"], "anthropic.messages.stream")
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
        self.assertTrue(request.headers["User-Agent"].startswith("MOTO-Autonomous-ASI/"))
        self.assertIn(CLAUDE_KEY, request.headers.get("authorization", "") + request.headers.get("x-api-key", ""))
        body = self.body()
        self.assertTrue(body["stream"])
        self.assertEqual(body["system"], "be exact")
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "xhigh"})
        self.assertEqual(body["max_tokens"], 4000)
        # Current Claude models reject sampling parameters, so none is ever sent.
        for sampling in ("temperature", "top_p", "top_k"):
            self.assertNotIn(sampling, body)
        self.assertNotIn("tools", body)
        self.assertNotIn("cache_control", body)

    async def test_auto_effort_uses_the_model_default_and_output_cap_is_respected(self) -> None:
        self.use(lambda request: messages_response())
        await self.complete(reasoning_effort="auto", max_tokens=500_000)
        body = self.body()
        self.assertNotIn("output_config", body)
        self.assertEqual(body["max_tokens"], 128_000)

    async def test_models_without_adaptive_thinking_or_effort_get_neither(self) -> None:
        self.use(lambda request: messages_response(model="claude-haiku-4-5"))
        await self.complete(model="claude-haiku-4-5", reasoning_effort="max", max_tokens=90_000)
        body = self.body()
        self.assertNotIn("thinking", body)
        self.assertNotIn("output_config", body)
        self.assertEqual(body["max_tokens"], 64_000)

    async def test_usage_counts_cached_input_and_truncation_and_refusal_keep_their_meaning(self) -> None:
        responses = [
            messages_response("partial", stop_reason="max_tokens", usage={
                "input_tokens": 3, "output_tokens": 4,
                "cache_read_input_tokens": 10, "cache_creation_input_tokens": 2,
            }),
            messages_response("", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}),
        ]
        self.use(lambda request: responses.pop(0))
        truncated = await self.complete()
        self.assertEqual(truncated["choices"][0]["finish_reason"], "length")
        self.assertEqual(truncated["usage"]["prompt_tokens"], 15)
        self.assertEqual(truncated["usage"]["total_tokens"], 19)
        self.assertEqual(truncated["usage"]["prompt_tokens_details"], {"cached_tokens": 10, "cache_write_tokens": 2})
        # A refusal is a model outcome, not a provider failure, and no other model answers.
        refused = await self.complete()
        self.assertEqual(refused["choices"][0]["finish_reason"], "content_filter")
        self.assertEqual(refused["choices"][0]["message"]["content"], "")
        self.assertEqual(refused["model"], "claude-opus-5-5")

    async def test_tool_turn_is_replayed_exactly_and_a_tool_free_follow_up_still_names_its_tools(self) -> None:
        responses = [
            messages_response(
                "", stop_reason="tool_use", thinking_signature="sig-1",
                tool_use={"id": "toolu_1", "name": "lookup", "input": {"q": 7}},
            ),
            messages_response("42"),
        ]
        self.use(lambda request: responses.pop(0))
        first = await self.complete(tools=[LOOKUP_TOOL])
        call = first["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(first["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(call["function"], {"name": "lookup", "arguments": '{"q": 7}'})
        first_body = self.body(0)
        self.assertEqual(first_body["tools"][0]["name"], "lookup")
        self.assertEqual(first_body["tools"][0]["input_schema"], LOOKUP_TOOL["function"]["parameters"])
        self.assertNotIn("tool_choice", first_body)
        self.assertEqual(first_body["cache_control"], {"type": "ephemeral"})

        history = [
            *USER,
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "seven"},
        ]
        # The caller's budget is spent, so it offers no tools on the final turn.
        second = await self.complete(messages=history, tools=None)
        self.assertEqual(second["choices"][0]["message"]["content"], "42")
        body = self.body(1)
        assistant = body["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0], {"type": "thinking", "thinking": "", "signature": "sig-1"})
        self.assertEqual(assistant["content"][1]["type"], "tool_use")
        self.assertEqual(assistant["content"][1]["input"], {"q": 7})
        self.assertEqual(body["messages"][2]["content"], [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "seven"},
        ])
        self.assertEqual(body["tools"][0]["input_schema"], LOOKUP_TOOL["function"]["parameters"])
        self.assertEqual(body["tool_choice"], {"type": "none"})

    async def test_rejected_replayed_thinking_is_retried_once_without_it(self) -> None:
        signature_error = messages_error(
            400, "invalid_request_error",
            "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.",
        )
        responses = [
            messages_response("", stop_reason="tool_use", thinking_signature="sig-1",
                              tool_use={"id": "toolu_1", "name": "lookup", "input": {}}),
            signature_error,
            messages_response("done"),
        ]
        self.use(lambda request: responses.pop(0))
        first = await self.complete(tools=[LOOKUP_TOOL])
        call = first["choices"][0]["message"]["tool_calls"][0]
        history = [
            *USER,
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
        ]
        result = await self.complete(messages=history, tools=[LOOKUP_TOOL])
        self.assertEqual(result["choices"][0]["message"]["content"], "done")
        self.assertEqual(self.body(1)["messages"][1]["content"][0]["type"], "thinking")
        retried = self.body(2)["messages"][1]["content"]
        self.assertEqual([block["type"] for block in retried], ["tool_use"])
        self.wait.assert_not_awaited()


class ClaudeApiRecoveryTests(ClaudeHarness):
    async def test_overload_and_rate_limit_retry_through_the_shared_wait(self) -> None:
        responses = [
            messages_error(529, "overloaded_error", "Overloaded"),
            messages_error(429, "rate_limit_error", "Too many requests", **{"retry-after": "3"}),
            messages_response("recovered"),
        ]
        self.use(lambda request: responses.pop(0))
        result = await self.complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(self.wait.await_count, 2)
        first, second = (call.args[0] for call in self.wait.await_args_list)
        self.assertEqual((first["provider"], first["reason"], first["model"]), ("claude_api", "transient_http_error", "claude-opus-5-5"))
        self.assertGreaterEqual(second["retry_after_seconds"], 3)

    async def test_exhausted_transient_failures_stay_transient_for_the_workflow(self) -> None:
        self.use(lambda request: messages_error(503, "api_error", "service unavailable"))
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.complete()
        self.assertIn("Anthropic Claude connection failed after retries", str(ctx.exception))
        self.assertTrue(is_transient_model_call_error(ctx.exception))
        self.assertEqual(len(self.requests), CLAUDE_API_SETTINGS.max_retries)

    async def test_a_stream_that_ends_without_a_stop_reason_is_retried(self) -> None:
        responses = [messages_response("half", stop_reason=None), messages_response("whole")]
        self.use(lambda request: responses.pop(0))
        result = await self.complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "whole")
        self.assertEqual(self.wait.await_args.args[0]["reason"], "stream_incomplete")

    async def test_an_error_inside_a_started_stream_is_classified_by_its_type(self) -> None:
        # The stream already answered HTTP 200, so only the error type says what happened.
        responses = [
            messages_stream_error("overloaded_error", "Overloaded"),
            messages_stream_error("api_error", "Internal server error"),
            messages_response("recovered"),
        ]
        self.use(lambda request: responses.pop(0))
        result = await self.complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(
            [call.args[0]["reason"] for call in self.wait.await_args_list],
            ["transient_http_error", "transient_http_error"],
        )

        self.wait.reset_mock()
        self.requests.clear()
        self.use(lambda request: messages_stream_error("invalid_request_error", "thinking: unexpected value"))
        with self.assertRaises(SubscriptionParameterRejectedError):
            await self.complete()
        self.assertEqual(len(self.requests), 1)
        self.wait.assert_not_awaited()

    async def test_an_unreadable_reply_is_retried_and_stays_transient_when_it_persists(self) -> None:
        unreadable = (
            httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>proxy sign-in</html>"),
            httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b""),
            httpx.Response(200, headers={"content-type": "text/event-stream"},
                           content=b"event: message_start\ndata: {not json}\n\n"),
        )
        for reply in unreadable:
            with self.subTest(content_type=reply.headers["content-type"], size=len(reply.content)):
                self.wait.reset_mock()
                responses = [reply, messages_response("recovered")]
                self.use(lambda request, responses=responses: responses.pop(0))
                result = await self.complete()
                self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
                self.assertEqual(self.wait.await_args.args[0]["reason"], "stream_incomplete")

        self.use(lambda request: unreadable[0])
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.complete()
        self.assertIn("connection failed after retries", str(ctx.exception))
        self.assertTrue(is_transient_model_call_error(ctx.exception))
        self.assertNotIn("proxy sign-in", str(ctx.exception))

    async def test_definitive_rejections_are_typed_and_never_retried(self) -> None:
        cases = (
            (messages_error(401, "authentication_error", "invalid x-api-key"), SubscriptionAuthError, "authentication_error"),
            (messages_error(403, "permission_error", "not allowed"), SubscriptionEntitlementError, "permission_denied"),
            (messages_error(404, "not_found_error", "model: claude-nope"), SubscriptionEntitlementError, "model_not_available"),
            (messages_error(402, "billing_error", "payment required"), SubscriptionEntitlementError, "billing_error"),
            (messages_error(400, "invalid_request_error", "Your credit balance is too low to access the Claude API."),
             SubscriptionEntitlementError, "billing_error"),
            (messages_error(400, "invalid_request_error", "your organization must have data retention enabled."),
             SubscriptionEntitlementError, "data_retention_required"),
            (messages_error(400, "invalid_request_error", "prompt is too long: 1200000 tokens > 1000000 maximum"),
             SubscriptionContextLimitError, "context_length_exceeded"),
            (messages_error(413, "request_too_large", "Request exceeds the maximum allowed number of bytes."),
             SubscriptionContextLimitError, "context_length_exceeded"),
            (messages_error(400, "invalid_request_error", "output_config.effort: unsupported value"),
             SubscriptionParameterRejectedError, "invalid_request_parameter"),
        )
        for response, error_cls, code in cases:
            with self.subTest(code=code, status=response.status_code):
                self.requests.clear()
                self.use(lambda request, response=response: response)
                with self.assertRaises(error_cls) as ctx:
                    await self.complete()
                self.assertEqual(ctx.exception.error_code, code)
                self.assertEqual(ctx.exception.provider, "claude_api")
                self.assertEqual(len(self.requests), 1)
                self.assertNotIn(CLAUDE_KEY, str(ctx.exception))
        self.wait.assert_not_awaited()

    async def test_context_rejection_carries_the_shared_overflow_marker(self) -> None:
        self.use(lambda request: messages_error(400, "invalid_request_error", "prompt is too long"))
        with self.assertRaises(SubscriptionContextLimitError) as ctx:
            await self.complete()
        self.assertTrue(is_provider_context_length_error(ctx.exception))

    async def test_parameter_rejection_names_a_moto_compatibility_problem(self) -> None:
        self.use(lambda request: messages_error(400, "invalid_request_error", "thinking: unexpected value"))
        with self.assertRaises(SubscriptionParameterRejectedError) as ctx:
            await self.complete()
        self.assertIn("MOTO compatibility problem", ctx.exception.terminal_guidance)
        self.assertIn("not a problem with your Anthropic Claude API key", ctx.exception.terminal_guidance)


class ClaudeApiModelListTests(ClaudeHarness):
    MODELS = {
        "data": [
            {
                "id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5",
                "created_at": "2026-09-01T00:00:00Z", "max_input_tokens": 1_000_000, "max_tokens": 128_000,
                "capabilities": {
                    "effort": {
                        "supported": True,
                        "low": {"supported": True}, "medium": {"supported": True}, "high": {"supported": True},
                        "xhigh": {"supported": True}, "max": {"supported": True},
                    },
                    "thinking": {"supported": True, "types": {
                        "adaptive": {"supported": True}, "enabled": {"supported": False},
                    }},
                },
            },
            {
                "id": "claude-future-9", "type": "model", "display_name": "Claude Future 9",
                "created_at": "2026-11-01T00:00:00Z", "max_input_tokens": 2_000_000, "max_tokens": 256_000,
                "capabilities": {
                    "effort": {
                        "supported": True,
                        "low": {"supported": True}, "medium": {"supported": True}, "high": {"supported": True},
                        "max": {"supported": False},
                    },
                    "thinking": {"supported": True, "types": {
                        "adaptive": {"supported": True}, "enabled": {"supported": False},
                    }},
                },
            },
        ],
        "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-future-9",
    }

    async def test_models_and_limits_come_from_the_models_api(self) -> None:
        self.use(lambda request: httpx.Response(200, json=self.MODELS))
        models = await self.client.list_models()
        self.assertEqual(self.requests[0].url.path, "/v1/models")
        by_id = {model["id"]: model for model in models}
        self.assertEqual(by_id["claude-opus-5-5"]["context_length"], 1_000_000)
        self.assertEqual(by_id["claude-opus-5-5"]["max_output_tokens"], 128_000)
        self.assertEqual(
            by_id["claude-opus-5-5"]["provider_metadata"]["reasoning_efforts"],
            ["auto", "max", "xhigh", "high", "medium", "low"],
        )
        future = by_id["claude-future-9"]
        self.assertEqual((future["name"], future["context_length"], future["max_output_tokens"]),
                         ("Claude Future 9", 2_000_000, 256_000))
        self.assertEqual(future["provider_metadata"]["reasoning_efforts"], ["auto", "high", "medium", "low"])
        # The provider's own metadata now shapes requests for a model MOTO has no table entry for.
        self.assertEqual(
            self.profile.traits_for("claude-future-9"),
            ClaudeModelTraits("Claude Future 9", 2_000_000, 256_000, ("low", "medium", "high"), True),
        )

    async def test_a_reply_that_is_not_a_model_list_is_a_typed_failure(self) -> None:
        self.use(lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html>proxy sign-in</html>",
        ))
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.client.list_models()
        self.assertNotIn("proxy sign-in", str(ctx.exception))

    async def test_candidate_validation_authenticates_without_changing_state(self) -> None:
        self.use(lambda request: httpx.Response(200, json=self.MODELS))
        validation = await self.client.validate_candidate("sk-claude-candidate-00000000")
        self.assertEqual(validation.region_id, "default")
        self.assertEqual(len(validation.models), 2)
        self.assertIn("sk-claude-candidate-00000000", self.requests[0].headers.get("authorization", "")
                      + self.requests[0].headers.get("x-api-key", ""))
        self.assertEqual(self.client._api_key, CLAUDE_KEY)
        self.assertEqual(self.profile.live_traits, {})

    async def test_a_rejected_candidate_key_is_an_auth_error_with_repair_guidance(self) -> None:
        self.use(lambda request: messages_error(401, "authentication_error", "invalid x-api-key"))
        with self.assertRaises(SubscriptionAuthError) as ctx:
            await self.client.validate_candidate("sk-claude-wrong-0000000000")
        self.assertIn("Claude Console", ctx.exception.terminal_guidance)
        self.assertNotIn("sk-claude-wrong-0000000000", str(ctx.exception))


class AmazonBedrockTests(ClaudeHarness):
    profile_factory = staticmethod(lambda: AmazonBedrockProfile(AMAZON_BEDROCK_SETTINGS))
    key = BEDROCK_KEY

    def setUp(self) -> None:
        super().setUp()
        self.client._region_id = "us-west-2"

    def test_claude_models_use_the_messages_api_and_others_use_chat_completions(self) -> None:
        wire = self.profile.wire_api_for_model
        self.assertEqual(wire("anthropic.claude-opus-5-5"), "anthropic_messages")
        self.assertEqual(wire("global.anthropic.claude-sonnet-5-5"), "anthropic_messages")
        self.assertEqual(wire("openai.gpt-oss-120b"), "chat_completions_stream")
        self.assertEqual(
            self.profile.messages_base_url(f"{MANTLE.format(region='eu-west-1')}/v1"),
            f"{MANTLE.format(region='eu-west-1')}/anthropic",
        )

    async def test_claude_request_goes_to_the_selected_regions_messages_route(self) -> None:
        self.use(lambda request: messages_response("Hello", model="anthropic.claude-opus-5-5"))
        result = await self.complete(model="anthropic.claude-opus-5-5", reasoning_effort="high")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
        request = self.requests[0]
        self.assertEqual(str(request.url), f"{MANTLE.format(region='us-west-2')}/anthropic/v1/messages")
        self.assertIn(BEDROCK_KEY, request.headers.get("authorization", "") + request.headers.get("x-api-key", ""))
        body = self.body()
        self.assertEqual(body["model"], "anthropic.claude-opus-5-5")
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", body)

    async def test_other_models_use_openai_compatible_chat_completions(self) -> None:
        self.use(lambda request: httpx.Response(200, content=chat_sse("Hi")))
        result = await self.complete(model="openai.gpt-oss-120b", reasoning_effort="max", temperature=0.2)
        self.assertEqual(result["choices"][0]["message"]["content"], "Hi")
        self.assertEqual(result["usage"]["total_tokens"], 7)
        request = self.requests[0]
        self.assertEqual(str(request.url), f"{MANTLE.format(region='us-west-2')}/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], f"Bearer {BEDROCK_KEY}")
        body = self.body()
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 4000)
        # Claude effort levels are not sent to models that are not Claude.
        self.assertNotIn("reasoning", body)
        self.assertNotIn("reasoning_effort", body)

    async def test_a_model_served_on_the_other_route_is_retried_there_and_remembered(self) -> None:
        wrong_route = httpx.Response(400, json={"error": {
            "type": "invalid_request_error",
            "message": "model `openai.gpt-5.6-sol` isn't supported on this route",
        }})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/chat/completions":
                return wrong_route
            return httpx.Response(200, content=chat_sse("routed"))

        self.use(handler)
        for _ in range(2):
            result = await self.complete(model="openai.gpt-5.6-sol")
            self.assertEqual(result["choices"][0]["message"]["content"], "routed")
        self.assertEqual(
            [request.url.path for request in self.requests],
            ["/v1/chat/completions", "/openai/v1/chat/completions", "/openai/v1/chat/completions"],
        )
        self.wait.assert_not_awaited()

    async def test_model_list_adds_documented_claude_models_and_limits(self) -> None:
        listed = {"data": [{"id": "openai.gpt-oss-120b"}, {"id": "anthropic.claude-sonnet-5-5"}]}
        self.use(lambda request: httpx.Response(200, json=listed))
        models = await self.client.list_models()
        self.assertEqual(str(self.requests[0].url), f"{MANTLE.format(region='us-west-2')}/v1/models")
        by_id = {model["id"]: model for model in models}
        self.assertEqual(set(by_id), {"openai.gpt-oss-120b", "anthropic.claude-sonnet-5-5"})
        sonnet = by_id["anthropic.claude-sonnet-5-5"]
        self.assertEqual((sonnet["name"], sonnet["context_length"], sonnet["max_output_tokens"]),
                         ("Claude Sonnet 5.5", 1_000_000, 128_000))
        # A model MOTO has no published limits for gets none invented.
        self.assertNotIn("context_length", by_id["openai.gpt-oss-120b"])
        self.assertNotIn("max_output_tokens", by_id["openai.gpt-oss-120b"])

        self.use(lambda request: httpx.Response(200, json={"data": [{"id": "openai.gpt-oss-120b"}]}))
        models = await self.client.list_models()
        ids = [model["id"] for model in models]
        self.assertEqual(ids[0], "anthropic.claude-opus-5-5")
        self.assertIn("openai.gpt-oss-120b", ids)

    async def test_claude_models_older_than_4_7_are_neither_listed_nor_sent(self) -> None:
        # Bedrock serves the Messages API version MOTO uses only for Claude 4.7 and later.
        listed = {"data": [
            {"id": "anthropic.claude-haiku-4-5"},
            {"id": "anthropic.claude-opus-4-6"},
            {"id": "anthropic.claude-opus-4-8"},
        ]}
        self.use(lambda request: httpx.Response(200, json=listed))
        models = await self.client.list_models()
        self.assertEqual([model["id"] for model in models], ["anthropic.claude-opus-4-8"])
        self.assertNotIn(
            "anthropic.claude-haiku-4-5",
            [model.model_id for model in AMAZON_BEDROCK_SETTINGS.known_models],
        )

        self.requests.clear()
        with self.assertRaises(SubscriptionEntitlementError) as ctx:
            await self.complete(model="anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(ctx.exception.error_code, "model_not_supported")
        self.assertIn("4.7", ctx.exception.terminal_guidance)
        self.assertEqual(self.requests, [])
        # The same model stays usable on the Claude API, which accepts that version for it.
        self.assertEqual(ClaudeApiProfile(CLAUDE_API_SETTINGS).unsupported_model_guidance("claude-haiku-4-5"), "")

    async def test_the_user_names_the_region_and_only_that_region_is_contacted(self) -> None:
        self.use(lambda request: httpx.Response(200, json={"data": [{"id": "anthropic.claude-opus-5-5"}]}))
        validation = await self.client.validate_candidate("candidate-key-000000000000", region_id="eu-central-1")
        self.assertEqual(validation.region_id, "eu-central-1")
        self.assertEqual([str(request.url) for request in self.requests],
                         [f"{MANTLE.format(region='eu-central-1')}/v1/models"])
        self.assertEqual(self.client._region_id, "us-west-2")
        with self.assertRaises(ValueError):
            await self.client.validate_candidate("candidate-key-000000000000", region_id="mars-north-1")

    async def test_saving_records_the_region_and_status_lists_the_choices(self) -> None:
        with mock.patch(f"{_CLIENT_MODULE}.store_subscription_api_key"), mock.patch(
            f"{_CLIENT_MODULE}.load_subscription_api_key", return_value="new-key-000000000000",
        ):
            self.client.set_api_key("new-key-000000000000", region_id="ap-northeast-1")
            status = await self.client.status()
        self.assertEqual(self.client.base_url, f"{MANTLE.format(region='ap-northeast-1')}/v1")
        self.assertEqual(status["region"], "ap-northeast-1")
        self.assertEqual(status["region_selection"], "user")
        self.assertIn({"id": "us-east-1", "label": "US East (N. Virginia) - us-east-1"}, status["region_options"])
        fresh = SubscriptionApiKeyClient(AmazonBedrockProfile(AMAZON_BEDROCK_SETTINGS))
        self.assertEqual(fresh.base_url, f"{MANTLE.format(region='ap-northeast-1')}/v1")

    async def test_an_unrecorded_region_follows_the_aws_environment(self) -> None:
        self.client._region_id = None
        with mock.patch.dict("os.environ", {"AWS_REGION": "eu-west-2"}):
            self.assertEqual(self.client.base_url, f"{MANTLE.format(region='eu-west-2')}/v1")
            self.assertEqual((await self.client.status())["default_region"], "eu-west-2")
        self.assertEqual(self.client.base_url, f"{MANTLE.format(region='us-east-1')}/v1")

    async def test_bedrock_rejections_carry_bedrock_repair_guidance(self) -> None:
        self.use(lambda request: messages_error(401, "authentication_error", "The security token is expired"))
        with self.assertRaises(SubscriptionAuthError) as ctx:
            await self.complete(model="anthropic.claude-opus-5-5")
        self.assertIn("expire within 12 hours", ctx.exception.terminal_guidance)

        self.use(lambda request: httpx.Response(403, json={"error": {"message": "access denied to model"}}))
        with self.assertRaises(SubscriptionEntitlementError) as ctx:
            await self.complete(model="openai.gpt-oss-120b")
        self.assertEqual(ctx.exception.error_code, "permission_denied")
        self.assertIn("Amazon Bedrock console", ctx.exception.terminal_guidance)

    async def test_a_missing_model_list_route_is_not_reported_as_a_missing_model(self) -> None:
        self.use(lambda request: httpx.Response(404, json={"error": {"message": "not found"}}))
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.client.list_models()
        self.assertNotIsInstance(ctx.exception, SubscriptionEntitlementError)


class RegionKeyRouteTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._patches = [mock.patch.object(system_config, "generic_mode", False)]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()

    @staticmethod
    def _endpoint(slug: str):
        for route in cloud_access_route.router.routes:
            if route.path == f"/api/cloud-access/{slug}/api-key" and "POST" in route.methods:
                return route.endpoint
        raise AssertionError(slug)

    async def test_the_key_route_passes_the_region_to_validation_and_storage(self) -> None:
        client = get_api_key_subscription_client("amazon_bedrock")
        validation = mock.Mock(region_id="eu-west-1", models=[{"id": "anthropic.claude-opus-5-5"}])
        with (
            mock.patch.object(client, "validate_candidate", mock.AsyncMock(return_value=validation)) as validate,
            mock.patch.object(client, "set_api_key") as set_api_key,
            mock.patch.object(client, "status", mock.AsyncMock(return_value={"configured": True})),
            mock.patch.object(cloud_access_route.api_client_manager, "reset_provider_fallbacks", mock.AsyncMock()),
        ):
            result = await self._endpoint("amazon-bedrock")(
                cloud_access_route.SubscriptionApiKeyRequest(api_key=BEDROCK_KEY, region="eu-west-1")
            )
        validate.assert_awaited_once_with(BEDROCK_KEY, region_id="eu-west-1")
        set_api_key.assert_called_once_with(BEDROCK_KEY, region_id="eu-west-1")
        self.assertEqual(result["provider"], "amazon_bedrock")

    async def test_an_unknown_region_is_a_client_error(self) -> None:
        client = get_api_key_subscription_client("amazon_bedrock")
        with mock.patch.object(client, "set_api_key") as set_api_key:
            with self.assertRaises(HTTPException) as ctx:
                await self._endpoint("amazon-bedrock")(
                    cloud_access_route.SubscriptionApiKeyRequest(api_key=BEDROCK_KEY, region="mars-north-1")
                )
        self.assertEqual(ctx.exception.status_code, 400)
        set_api_key.assert_not_called()


class ClaudeRoleConfigTests(TestCase):
    def test_both_providers_are_valid_role_providers_and_accept_max_effort(self) -> None:
        for provider in ("claude_api", "amazon_bedrock"):
            config = ModelConfig(
                provider=provider,
                model_id="claude-opus-5-5",
                openrouter_reasoning_effort="max",
                context_window=1_000_000,
                max_output_tokens=64_000,
            )
            self.assertEqual(config.provider, provider)
            self.assertEqual(config.openrouter_reasoning_effort, "max")
