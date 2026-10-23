"""OpenRouterClient streamed Chat Completions end to end over ``httpx.MockTransport``.

Completions are requested with ``stream: true`` and reassembled into the
non-streaming response shape. Pre-commit HTTP errors and JSON bodies keep the
original classification; a stream that never finishes retries as a transient
route failure, while a drop after the finish reason keeps the confirmed answer.
"""
import json
from typing import Callable, Iterable, List, Optional
from unittest import IsolatedAsyncioTestCase, mock

import httpx

from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.openrouter_client import (
    CreditExhaustionError,
    OpenRouterClient,
    OpenRouterPrivacyPolicyError,
    RateLimitError,
)
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteError
from backend.shared.response_extraction import (
    choice_hit_output_limit,
    drop_unfinished_thinking,
    extract_response_text,
)

COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "sk-or-test-streaming-key"
PAID_MODEL = "vendor/test-paid"
FREE_MODEL = "vendor/test-stream:free"
MESSAGES = [{"role": "user", "content": "hello"}]
STREAM_ID = "gen-stream-1"
STREAM_HOST = "TestHost"
KEEPALIVE = b": OPENROUTER PROCESSING\n\n"


class ChunkedByteStream(httpx.AsyncByteStream):
    """Yield fixed byte chunks, then optionally fail like a dropped connection."""

    def __init__(self, chunks: Iterable[bytes], *, fail_with: Optional[BaseException] = None) -> None:
        self._chunks = list(chunks)
        self._fail_with = fail_with

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._fail_with is not None:
            raise self._fail_with


def sse_event(payload) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode("utf-8")


def chunk(
    delta: Optional[dict] = None,
    *,
    finish_reason: Optional[str] = None,
    native_finish_reason: Optional[str] = None,
    **top,
) -> dict:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    if native_finish_reason is not None:
        choice["native_finish_reason"] = native_finish_reason
    return {
        "id": STREAM_ID,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "vendor/upstream-model",
        "provider": STREAM_HOST,
        "choices": [choice],
        **top,
    }


def answer_chunks(text: str = "Hello", *, finish_reason: str = "stop") -> List[dict]:
    return [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": text}),
        chunk(finish_reason=finish_reason),
    ]


def stream_response(
    *payloads,
    done: bool = True,
    fail_with: Optional[BaseException] = None,
) -> Callable[[], httpx.Response]:
    """Factory for a fresh ``text/event-stream`` response with an unread body."""

    def build() -> httpx.Response:
        chunks = [KEEPALIVE] + [sse_event(payload) for payload in payloads]
        if done:
            chunks.append(sse_event("[DONE]"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedByteStream(chunks, fail_with=fail_with),
        )

    return build


def json_response(status: int, body) -> Callable[[], httpx.Response]:
    """Factory for a JSON response whose body still has to be read from the stream."""

    def build() -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "application/json"},
            stream=ChunkedByteStream([json.dumps(body).encode("utf-8")]),
        )

    return build


class OpenRouterStreamingHarness(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved_rate_limits = dict(OpenRouterClient._rate_limited_models)
        OpenRouterClient._rate_limited_models.clear()
        self.requests: List[httpx.Request] = []
        self.retry_callback = mock.AsyncMock()

    async def asyncSetUp(self) -> None:
        self.client = OpenRouterClient(API_KEY)
        # Close the real transport immediately; every test serves OpenRouter itself.
        await self.client.client.aclose()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    def tearDown(self) -> None:
        OpenRouterClient._rate_limited_models.clear()
        OpenRouterClient._rate_limited_models.update(self._saved_rate_limits)

    def serve(self, *factories: Callable[[], httpx.Response]) -> None:
        pending = list(factories)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if not pending:
                raise AssertionError(f"Unexpected extra OpenRouter request #{len(self.requests)}")
            return pending.pop(0)()

        self.client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def payload(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)

    def retry_payloads(self) -> List[dict]:
        return [call.args[0] for call in self.retry_callback.await_args_list]

    async def complete(self, model: str = PAID_MODEL, *, use_callback: bool = True, **kwargs):
        if use_callback:
            kwargs["retry_callback"] = self.retry_callback
        return await self.client.generate_completion(
            model=model,
            messages=MESSAGES,
            max_tokens=64,
            **kwargs,
        )


class StreamedSuccessTests(OpenRouterStreamingHarness):
    async def test_request_asks_for_a_stream(self) -> None:
        self.serve(stream_response(*answer_chunks()))

        await self.complete()

        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), COMPLETIONS_URL)
        self.assertEqual(request.headers["Authorization"], f"Bearer {API_KEY}")
        self.assertEqual(request.headers["X-Title"], OpenRouterClient.APP_TITLE)
        body = self.payload()
        self.assertIs(body["stream"], True)
        self.assertEqual(body["model"], PAID_MODEL)
        self.assertEqual(body["messages"], MESSAGES)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["provider"], {"ignore": list(OpenRouterClient.AUTO_IGNORED_PROVIDERS)})

    async def test_streamed_answer_returns_the_non_streaming_shape(self) -> None:
        self.serve(
            stream_response(
                chunk({"role": "assistant", "content": ""}),
                chunk({"content": "Hel"}),
                chunk({"content": "lo"}),
                chunk(finish_reason="stop", native_finish_reason="end_turn"),
            )
        )

        result = await self.complete()

        self.assertEqual(result["id"], STREAM_ID)
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["created"], 1700000000)
        self.assertEqual(result["model"], "vendor/upstream-model")
        self.assertEqual(result["provider"], STREAM_HOST)
        self.assertEqual(
            result["choices"],
            [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                }
            ],
        )
        self.assertEqual(extract_response_text(result), "Hello")
        self.assertNotIn("_moto_openrouter_provider_auto_fallback", result)
        self.retry_callback.assert_not_awaited()

    async def test_missing_stream_model_defaults_to_the_requested_model(self) -> None:
        self.serve(stream_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}))

        result = await self.complete()

        self.assertEqual(result["model"], PAID_MODEL)
        self.assertEqual(extract_response_text(result), "ok")

    async def test_reasoning_only_stream_uses_the_reasoning_fallback(self) -> None:
        self.serve(
            stream_response(
                chunk({"role": "assistant", "reasoning": "The answer "}),
                chunk({"reasoning": "is 4"}),
                chunk(finish_reason="stop"),
            )
        )

        result = await self.complete()

        message = result["choices"][0]["message"]
        self.assertIsInstance(message["content"], str)
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning"], "The answer is 4")
        self.assertEqual(extract_response_text(result), "The answer is 4")

    async def test_output_limit_finish_reason_is_preserved(self) -> None:
        self.serve(
            stream_response(
                chunk({"reasoning": "still thinking"}),
                chunk(finish_reason="length", native_finish_reason="max_tokens"),
            ),
            stream_response(
                chunk({"content": "partial answer"}),
                chunk(finish_reason="length"),
            ),
        )

        thinking_only = await self.complete()
        choice = thinking_only["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(choice["native_finish_reason"], "max_tokens")
        self.assertTrue(choice_hit_output_limit(choice))
        self.assertEqual(extract_response_text(thinking_only), "")
        drop_unfinished_thinking(thinking_only)
        self.assertNotIn("reasoning", choice["message"])

        partial = await self.complete()
        self.assertTrue(choice_hit_output_limit(partial["choices"][0]))
        self.assertEqual(extract_response_text(partial), "partial answer")
        self.retry_callback.assert_not_awaited()

    async def test_streamed_tool_calls_are_reassembled(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {"name": "wolfram_alpha_query", "parameters": {"type": "object"}},
            }
        ]
        self.serve(
            stream_response(
                chunk({"role": "assistant", "content": None}),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "wolfram_alpha_query", "arguments": ""},
                            }
                        ]
                    }
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": '}}]}),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"2+2"}'}}]}),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "wolfram_alpha_query", "arguments": '{"query": "3+3"}'},
                            }
                        ]
                    }
                ),
                chunk(finish_reason="tool_calls"),
            )
        )

        result = await self.complete(tools=tools, tool_choice="auto")

        body = self.payload()
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "")
        self.assertEqual(
            choice["message"]["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "wolfram_alpha_query", "arguments": '{"query": "2+2"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "wolfram_alpha_query", "arguments": '{"query": "3+3"}'},
                },
            ],
        )
        for call in choice["message"]["tool_calls"]:
            self.assertNotIn("index", call)
            self.assertIsInstance(json.loads(call["function"]["arguments"]), dict)

    async def test_usage_is_passed_through(self) -> None:
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "cost": 0.000123,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        self.serve(
            stream_response(
                chunk({"content": "Hi"}),
                chunk(finish_reason="stop"),
                chunk(finish_reason="stop", usage=usage),
            )
        )

        result = await self.complete()

        self.assertEqual(result["usage"], usage)
        self.assertEqual(len(result["choices"]), 1)
        self.assertEqual(extract_response_text(result), "Hi")

    async def test_json_completion_body_is_still_accepted(self) -> None:
        plain = {"id": "gen-json-ok", "choices": [{"index": 0, "message": {"content": "plain"}, "finish_reason": "stop"}]}
        self.serve(json_response(200, plain))

        result = await self.complete()

        self.assertEqual(result, plain)
        self.retry_callback.assert_not_awaited()


class IncompleteStreamTests(OpenRouterStreamingHarness):
    async def test_stream_without_finish_reason_retries_then_raises_transient_route_error(self) -> None:
        unfinished = (chunk({"role": "assistant", "content": ""}), chunk({"content": "partial"}))
        self.serve(
            stream_response(*unfinished),
            stream_response(*unfinished, done=False),
            stream_response(*unfinished),
        )

        with self.assertRaises(ProviderRouteError) as raised:
            await self.complete()

        self.assertEqual(len(self.requests), 3)
        payloads = self.retry_payloads()
        self.assertEqual(len(payloads), 2)
        self.assertEqual([payload["retry_attempt"] for payload in payloads], [1, 2])
        for payload in payloads:
            self.assertEqual(payload["reason"], "connection_error")
            self.assertEqual(payload["error_summary"], "OpenRouter stream ended before the completion finished.")
            self.assertEqual(payload["upstream_host_provider"], STREAM_HOST)
            self.assertEqual(payload["provider_request_id"], STREAM_ID)
            self.assertEqual(payload["provider"], "openrouter")
            self.assertEqual(payload["model"], PAID_MODEL)
        error = raised.exception
        self.assertIsInstance(error.cause, httpx.RemoteProtocolError)
        self.assertTrue(
            str(error).startswith("OpenRouter stream ended before the completion finished after 3 attempts.")
        )
        self.assertNotIn("partial", str(error))
        self.assertEqual(error.route.upstream_host_provider, STREAM_HOST)
        self.assertEqual(error.route.provider_request_id, STREAM_ID)
        self.assertTrue(is_transient_model_call_error(error))
        self.assertFalse(is_non_retryable_model_error(error))

    async def test_empty_stream_is_treated_as_unfinished(self) -> None:
        self.serve(stream_response(), stream_response(*answer_chunks("recovered")))

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payloads = self.retry_payloads()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["reason"], "connection_error")
        self.assertEqual(payloads[0]["upstream_host_provider"], "")
        self.assertEqual(payloads[0]["provider_request_id"], "")

    async def test_unfinished_stream_recovers_on_retry(self) -> None:
        self.serve(
            stream_response(chunk({"content": "partial"})),
            stream_response(*answer_chunks("complete answer")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "complete answer")
        self.assertEqual(len(self.requests), 2)
        self.assertIs(self.payload(1)["stream"], True)
        self.assertEqual([payload["reason"] for payload in self.retry_payloads()], ["connection_error"])

    async def test_direct_retry_wait_is_used_without_a_callback(self) -> None:
        self.serve(
            stream_response(chunk({"content": "partial"})),
            stream_response(*answer_chunks("complete answer")),
        )

        with mock.patch(
            "backend.shared.openrouter_client.assured_retry_wait",
            new=mock.AsyncMock(return_value=True),
        ) as retry_wait:
            result = await self.complete(use_callback=False)

        self.assertEqual(extract_response_text(result), "complete answer")
        retry_wait.assert_awaited_once()
        self.assertEqual(retry_wait.await_args.kwargs["role_id"], "openrouter_direct")
        payload = retry_wait.await_args.args[0]
        self.assertEqual(payload["reason"], "connection_error")
        self.assertEqual(payload["error_summary"], "OpenRouter stream ended before the completion finished.")

    async def test_drop_after_finish_reason_keeps_the_answer(self) -> None:
        self.serve(
            stream_response(
                chunk({"content": "Kept"}),
                chunk(finish_reason="stop"),
                done=False,
                fail_with=httpx.ReadError("connection dropped after the finish"),
            )
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "Kept")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(self.requests), 1)
        self.retry_callback.assert_not_awaited()

    async def test_drop_before_finish_reason_retries_as_connection_error(self) -> None:
        self.serve(
            stream_response(
                chunk({"content": "par"}),
                done=False,
                fail_with=httpx.ReadError("connection dropped mid-answer"),
            ),
            stream_response(*answer_chunks("second try")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "second try")
        self.assertEqual(len(self.requests), 2)
        payloads = self.retry_payloads()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["reason"], "connection_error")
        self.assertIn("ReadError", payloads[0]["error_summary"])
        # The host and request id already streamed stay in the structured route fields only.
        self.assertEqual(payloads[0]["upstream_host_provider"], STREAM_HOST)
        self.assertEqual(payloads[0]["provider_request_id"], STREAM_ID)
        self.assertNotIn(STREAM_HOST, payloads[0]["error_summary"])

    async def test_drop_after_an_error_event_keeps_the_error_classification(self) -> None:
        upstream_error = {"code": 504, "message": "A Timeout Occurred", "metadata": {"provider_name": "Nvidia"}}
        self.serve(
            stream_response(
                chunk(finish_reason="error", error=upstream_error),
                done=False,
                fail_with=httpx.ReadError("connection dropped after the error event"),
            ),
            stream_response(*answer_chunks("recovered")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payload = self.retry_payloads()[0]
        self.assertEqual(payload["reason"], "invalid_response")
        self.assertEqual(
            payload["error_summary"],
            "OpenRouter stream returned an error before the completion finished "
            "(upstream 504: A Timeout Occurred).",
        )
        self.assertEqual(payload["upstream_host_provider"], "Nvidia")

    async def test_unreadable_chunk_is_never_accepted_as_a_complete_answer(self) -> None:
        self.serve(
            stream_response(chunk({"content": "theorem foo : "}), '{"choices": [', *answer_chunks(" := by simp")),
            stream_response(*answer_chunks("intact")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "intact")
        payload = self.retry_payloads()[0]
        self.assertEqual(payload["reason"], "invalid_response")
        self.assertEqual(
            payload["error_summary"],
            "OpenRouter stream contained unreadable data, so part of the answer may be missing.",
        )

    async def test_repeated_drops_before_finish_reason_exhaust_as_transient_route_error(self) -> None:
        def dropped() -> Callable[[], httpx.Response]:
            return stream_response(
                chunk({"content": "par"}),
                done=False,
                fail_with=httpx.ReadError("connection dropped mid-answer"),
            )

        self.client.MAX_RETRIES = 2
        self.serve(dropped(), dropped())

        with self.assertRaises(ProviderRouteError) as raised:
            await self.complete()

        self.assertEqual(len(self.requests), 2)
        self.assertEqual([payload["reason"] for payload in self.retry_payloads()], ["connection_error"])
        self.assertIsInstance(raised.exception.cause, httpx.ReadError)
        self.assertTrue(is_transient_model_call_error(raised.exception))

    async def test_stream_error_event_retries_as_invalid_response(self) -> None:
        upstream_error = {"code": 502, "message": "Upstream overloaded", "metadata": {"provider_name": "FailingHost"}}
        self.serve(
            stream_response(
                chunk({"content": "par"}),
                chunk(finish_reason="error", error=upstream_error),
            ),
            stream_response(*answer_chunks("recovered")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payloads = self.retry_payloads()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["reason"], "invalid_response")
        self.assertEqual(
            payloads[0]["error_summary"],
            "OpenRouter stream returned an error before the completion finished "
            "(upstream 502: Upstream overloaded).",
        )
        self.assertEqual(payloads[0]["upstream_host_provider"], "FailingHost")
        self.assertEqual(payloads[0]["provider_request_id"], STREAM_ID)

    async def test_bare_finish_reason_error_retries_as_invalid_response(self) -> None:
        self.serve(
            stream_response(chunk({"content": "par"}), chunk(finish_reason="error")),
            stream_response(*answer_chunks("recovered")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payload = self.retry_payloads()[0]
        self.assertEqual(payload["reason"], "invalid_response")
        self.assertTrue(
            payload["error_summary"].startswith(
                "OpenRouter stream returned an error before the completion finished (upstream: "
            )
        )
        self.assertIn("finish_reason 'error'", payload["error_summary"])
        self.assertEqual(payload["upstream_host_provider"], STREAM_HOST)

    async def test_stream_error_exhaustion_is_a_transient_route_error(self) -> None:
        error_only = {
            "id": "gen-err-1",
            "error": {"code": 502, "message": "Upstream overloaded", "metadata": {"provider_name": "FailingHost"}},
        }
        self.client.MAX_RETRIES = 2
        self.serve(stream_response(error_only, done=False), stream_response(error_only))

        with self.assertRaises(ProviderRouteError) as raised:
            await self.complete()

        self.assertEqual(len(self.requests), 2)
        error = raised.exception
        self.assertTrue(
            str(error).startswith(
                "OpenRouter stream returned an error before the completion finished after 2 attempts."
            )
        )
        self.assertNotIn("Upstream overloaded", str(error))
        self.assertIsInstance(error.cause, httpx.ReadError)
        self.assertEqual(error.route.upstream_host_provider, "FailingHost")
        self.assertEqual(error.route.provider_request_id, "gen-err-1")
        self.assertTrue(is_transient_model_call_error(error))
        self.assertFalse(is_non_retryable_model_error(error))


class PreCommitErrorTests(OpenRouterStreamingHarness):
    async def test_402_is_credit_exhaustion(self) -> None:
        self.serve(json_response(402, {"error": {"code": 402, "message": "Insufficient credits"}}))

        with self.assertRaises(CreditExhaustionError):
            await self.complete()

        self.assertEqual(len(self.requests), 1)
        self.assertIs(self.payload()["stream"], True)
        self.retry_callback.assert_not_awaited()

    async def test_429_on_a_free_model_records_the_rate_limit(self) -> None:
        self.serve(json_response(429, {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}))

        with self.assertRaises(RateLimitError) as raised:
            await self.complete(FREE_MODEL)

        self.assertEqual(raised.exception.model, FREE_MODEL)
        self.assertIn(FREE_MODEL, OpenRouterClient._rate_limited_models)
        self.retry_callback.assert_not_awaited()

        with self.assertRaises(RateLimitError):
            await self.complete(FREE_MODEL)
        self.assertEqual(len(self.requests), 1)

    async def test_404_data_policy_is_a_privacy_policy_error(self) -> None:
        self.serve(
            json_response(
                404,
                {
                    "error": {
                        "code": 404,
                        "message": (
                            "No endpoints found matching your data policy (Free model publication). "
                            "Configure: https://openrouter.ai/settings/privacy"
                        ),
                    }
                },
            )
        )

        with self.assertRaises(OpenRouterPrivacyPolicyError):
            await self.complete(FREE_MODEL)

        self.assertEqual(len(self.requests), 1)
        self.retry_callback.assert_not_awaited()

    async def test_400_context_length_is_typed(self) -> None:
        self.serve(
            json_response(
                400,
                {"error": {"code": 400, "message": "context_length_exceeded: prompt is too long"}},
            )
        )

        with self.assertRaises(ProviderContextLengthError) as raised:
            await self.complete(provider="PinnedHost")

        self.assertEqual(len(self.requests), 1)
        self.assertEqual(raised.exception.route.provider, "openrouter")
        self.assertEqual(raised.exception.route.host_provider, "PinnedHost")
        self.assertTrue(is_non_retryable_model_error(raised.exception))
        self.retry_callback.assert_not_awaited()

    async def test_server_error_before_streaming_retries_as_http_error(self) -> None:
        self.serve(
            json_response(
                503,
                {"id": "gen-503", "error": {"code": 503, "message": "upstream unavailable", "metadata": {"provider_name": "BusyHost"}}},
            ),
            stream_response(*answer_chunks("recovered")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payload = self.retry_payloads()[0]
        self.assertEqual(payload["reason"], "http_error")
        self.assertEqual(payload["upstream_host_provider"], "BusyHost")
        self.assertEqual(payload["provider_request_id"], "gen-503")

    async def test_json_error_only_body_keeps_the_missing_choices_summary(self) -> None:
        self.serve(
            json_response(
                200,
                {
                    "id": "gen-json-1",
                    "error": {"code": 504, "message": "A Timeout Occurred", "metadata": {"provider_name": "SlowHost"}},
                },
            ),
            stream_response(*answer_chunks("recovered")),
        )

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "recovered")
        payloads = self.retry_payloads()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["reason"], "invalid_response")
        self.assertEqual(
            payloads[0]["error_summary"],
            "OpenRouter returned a successful response without completion choices "
            "(upstream 504: A Timeout Occurred).",
        )
        self.assertEqual(payloads[0]["upstream_host_provider"], "SlowHost")
        self.assertEqual(payloads[0]["provider_request_id"], "gen-json-1")

    async def test_provider_auto_fallback_marks_the_streamed_success(self) -> None:
        self.serve(
            json_response(404, {"error": {"code": 404, "message": f"No endpoints found for {PAID_MODEL}."}}),
            stream_response(*answer_chunks("rerouted")),
        )

        result = await self.complete(provider="PinnedHost", allow_provider_auto_fallback=True)

        self.assertEqual(extract_response_text(result), "rerouted")
        self.assertEqual(
            result["_moto_openrouter_provider_auto_fallback"],
            {"from_provider": "PinnedHost"},
        )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.payload(0)["provider"], {"order": ["PinnedHost"], "allow_fallbacks": False})
        self.assertEqual(
            self.payload(1)["provider"],
            {"ignore": list(OpenRouterClient.AUTO_IGNORED_PROVIDERS)},
        )
        self.assertIs(self.payload(1)["stream"], True)
        self.retry_callback.assert_not_awaited()

    async def test_auto_routing_retry_still_runs_after_the_last_pinned_attempt(self) -> None:
        def unfinished() -> Callable[[], httpx.Response]:
            return stream_response(chunk({"content": "par"}), done=False)

        self.client.MAX_RETRIES = 2
        self.serve(
            unfinished(),
            json_response(404, {"error": {"code": 404, "message": f"No endpoints found for {PAID_MODEL}."}}),
            stream_response(*answer_chunks("rerouted")),
        )

        result = await self.complete(provider="PinnedHost", allow_provider_auto_fallback=True)

        self.assertEqual(extract_response_text(result), "rerouted")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.payload(2)["provider"], {"ignore": list(OpenRouterClient.AUTO_IGNORED_PROVIDERS)})
        self.assertEqual(
            result["_moto_openrouter_provider_auto_fallback"],
            {"from_provider": "PinnedHost"},
        )


class StreamBodyDrainTests(OpenRouterStreamingHarness):
    async def test_body_is_read_to_its_end_after_done(self) -> None:
        # Reading past [DONE] completes the chunked body so the connection can be reused.
        consumed: List[bytes] = []

        class RecordingStream(ChunkedByteStream):
            async def __aiter__(self):
                async for piece in super().__aiter__():
                    consumed.append(piece)
                    yield piece

        tail = b": trailing keepalive after done\n\n"
        chunks = [sse_event(payload) for payload in answer_chunks("drained")] + [sse_event("[DONE]"), tail]
        self.serve(lambda: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=RecordingStream(chunks),
        ))

        result = await self.complete()

        self.assertEqual(extract_response_text(result), "drained")
        self.assertEqual(consumed[-1], tail)


if __name__ == "__main__":
    import unittest

    unittest.main()
