"""Pure tests of SSE framing and Chat Completions stream reassembly.

``backend.shared.chat_completion_stream`` is provider-neutral: it frames
server-sent events and rebuilds the non-streaming response shape, reporting
provider errors as data so each client keeps its own classification policy.
"""
import json
import unittest
from typing import Iterable, List, Optional

import httpx

from backend.shared.chat_completion_stream import (
    DONE_SENTINEL,
    ChatCompletionStreamAccumulator,
    is_event_stream,
    iter_sse_data,
    split_sse_data,
)
from backend.shared.openrouter_client import _reported_upstream


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


def streamed_response(chunks: Iterable[bytes], *, fail_with: Optional[BaseException] = None) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=ChunkedByteStream(chunks, fail_with=fail_with),
    )


async def collect_events(chunks: Iterable[bytes]) -> List[str]:
    return [event async for event in iter_sse_data(streamed_response(chunks))]


def delta_event(delta: dict, *, index: Optional[int] = 0, finish_reason: Optional[str] = None, **top) -> str:
    choice: dict = {"delta": delta}
    if index is not None:
        choice["index"] = index
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return json.dumps({"choices": [choice], **top})


def finish_event(reason: str = "stop", *, index: int = 0, **top) -> str:
    return delta_event({}, index=index, finish_reason=reason, **top)


def feed_all(stream: ChatCompletionStreamAccumulator, *events: str) -> List[bool]:
    return [stream.feed(event) for event in events]


# Bodies exercised by both framers; the async framer must agree with the buffered one.
PARITY_BODIES = (
    ": OPENROUTER PROCESSING\n\n: OPENROUTER PROCESSING\n\ndata: {\"a\": 1}\n\ndata: [DONE]\n\n",
    "data: first\ndata: second\n: keepalive inside the event\ndata: third\n\n",
    "data: crlf one\r\n\r\ndata: crlf two\r\ndata: crlf three\r\n\r\n",
    "data: lone cr\r\rdata: second lone cr\r\r",
    "event: completion\nid: 42\nretry: 3000\ndata: payload\nunknown: ignored\n\nevent: ping\n\n",
    "data: complete\n\ndata: trailing without blank line",
    "data:no-space\n\n\n\n\ndata:  extra-space\n\n",
    'data: {"choices": [{"delta": {"content": "h\\u00e9llo \u2013 \u4e16\u754c \u00e9"}}]}\n\n',
    "",
)


def split_whole(body: bytes) -> List[bytes]:
    return [body] if body else []


def split_each_byte(body: bytes) -> List[bytes]:
    return [body[i:i + 1] for i in range(len(body))]


def split_three_bytes(body: bytes) -> List[bytes]:
    return [body[i:i + 3] for i in range(0, len(body), 3)]


def split_between_cr_and_lf(body: bytes) -> List[bytes]:
    parts: List[bytes] = []
    start = 0
    for position in range(len(body) - 1):
        if body[position:position + 1] == b"\r":
            parts.append(body[start:position + 1])
            start = position + 1
    parts.append(body[start:])
    return [part for part in parts if part]


SPLITTERS = (split_whole, split_each_byte, split_three_bytes, split_between_cr_and_lf)


class SplitSseDataTests(unittest.TestCase):
    def test_comment_lines_are_skipped(self) -> None:
        body = ": OPENROUTER PROCESSING\n\n: OPENROUTER PROCESSING\n\ndata: {\"a\": 1}\n\n"
        self.assertEqual(split_sse_data(body), ['{"a": 1}'])

    def test_comment_only_body_yields_no_events(self) -> None:
        self.assertEqual(split_sse_data(": OPENROUTER PROCESSING\n\n:\n\n"), [])
        self.assertEqual(split_sse_data(""), [])

    def test_comment_line_inside_an_event_does_not_split_it(self) -> None:
        self.assertEqual(split_sse_data("data: one\n: keepalive\ndata: two\n\n"), ["one\ntwo"])

    def test_multi_line_data_is_joined_with_newlines(self) -> None:
        body = "data: first\ndata: second\ndata: third\n\ndata: next\n\n"
        self.assertEqual(split_sse_data(body), ["first\nsecond\nthird", "next"])

    def test_crlf_and_lone_cr_line_endings(self) -> None:
        self.assertEqual(
            split_sse_data("data: a\r\n\r\ndata: b\r\ndata: c\r\n\r\n"),
            ["a", "b\nc"],
        )
        self.assertEqual(split_sse_data("data: a\r\rdata: b\r\r"), ["a", "b"])

    def test_trailing_event_without_blank_line_is_flushed(self) -> None:
        self.assertEqual(split_sse_data("data: a\n\ndata: b"), ["a", "b"])
        self.assertEqual(split_sse_data("data: only\n"), ["only"])
        self.assertEqual(split_sse_data("data: x\ndata: y"), ["x\ny"])

    def test_non_data_fields_are_ignored(self) -> None:
        body = "event: completion\nid: 42\nretry: 3000\ndata: payload\nunknown: x\n\nevent: ping\n\n"
        self.assertEqual(split_sse_data(body), ["payload"])

    def test_data_prefix_without_space(self) -> None:
        self.assertEqual(split_sse_data('data:{"x":1}\n\n'), ['{"x":1}'])

    def test_extra_blank_lines_between_events_produce_nothing(self) -> None:
        self.assertEqual(split_sse_data("\n\n\ndata: a\n\n\n\n\ndata: b\n\n\n"), ["a", "b"])

    def test_done_sentinel_is_an_ordinary_event(self) -> None:
        self.assertEqual(split_sse_data("data: {}\n\ndata: [DONE]\n\n"), ["{}", DONE_SENTINEL])


class IterSseDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_framer_matches_buffered_framer_for_any_chunking(self) -> None:
        for body in PARITY_BODIES:
            expected = split_sse_data(body)
            encoded = body.encode("utf-8")
            for splitter in SPLITTERS:
                with self.subTest(body=body, splitter=splitter.__name__):
                    self.assertEqual(await collect_events(splitter(encoded)), expected)

    async def test_events_are_yielded_before_the_stream_fails(self) -> None:
        response = streamed_response(
            [b"data: first\n\n", b"data: partial-without-terminator"],
            fail_with=httpx.ReadError("connection dropped"),
        )
        received: List[str] = []
        with self.assertRaises(httpx.ReadError):
            async for event in iter_sse_data(response):
                received.append(event)
        self.assertEqual(received, ["first"])

    async def test_unicode_line_separators_inside_json_do_not_split_events(self) -> None:
        # JSON leaves U+0085/U+2028/U+2029 unescaped; only CR, LF and CRLF end an SSE line.
        content = "a\u0085b c d"
        body = "data: " + json.dumps({"content": content}, ensure_ascii=False) + "\n\n"
        self.assertEqual([json.loads(event)["content"] for event in split_sse_data(body)], [content])
        for splitter in SPLITTERS:
            with self.subTest(splitter=splitter.__name__):
                events = await collect_events(splitter(body.encode("utf-8")))
                self.assertEqual([json.loads(event)["content"] for event in events], [content])

    async def test_split_multibyte_character_is_decoded(self) -> None:
        body = 'data: {"content": "\u4e16\u754c"}\n\n'.encode("utf-8")
        events = await collect_events(split_each_byte(body))
        self.assertEqual(json.loads(events[0])["content"], "\u4e16\u754c")


class IsEventStreamTests(unittest.TestCase):
    def test_content_type_detection(self) -> None:
        cases = (
            ({"content-type": "text/event-stream"}, True),
            ({"content-type": "text/event-stream; charset=utf-8"}, True),
            ({"content-type": "Text/Event-Stream"}, True),
            ({"content-type": "application/json"}, False),
            ({"content-type": "text/html; charset=utf-8"}, False),
            ({}, False),
        )
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertIs(is_event_stream(httpx.Response(200, headers=headers)), expected)


class AccumulatorContentTests(unittest.TestCase):
    def test_content_deltas_concatenate_and_finish_reason_completes(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event(
                {"role": "assistant", "content": ""},
                id="gen-1",
                model="vendor/model",
                created=1700000000,
                provider="HostA",
            ),
            delta_event({"content": "Hel"}),
            delta_event({"content": "lo"}),
        )
        self.assertFalse(stream.terminal)
        self.assertFalse(stream.failed)

        stream.feed(finish_event("stop"))

        self.assertTrue(stream.terminal)
        self.assertFalse(stream.failed)
        self.assertEqual(
            stream.result("default/model"),
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "vendor/model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "provider": "HostA",
            },
        )
        self.assertEqual(stream.valid_chunks, 4)
        self.assertEqual(stream.malformed_chunks, 0)

    def test_reasoning_and_reasoning_content_keep_their_own_keys(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"reasoning": "Think "}),
            delta_event({"reasoning": "harder", "reasoning_content": "alt "}),
            delta_event({"reasoning_content": "trace"}),
            delta_event({"content": "answer"}),
            finish_event(),
        )
        message = stream.result()["choices"][0]["message"]
        self.assertEqual(
            message,
            {
                "role": "assistant",
                "content": "answer",
                "reasoning": "Think harder",
                "reasoning_content": "alt trace",
            },
        )

    def test_only_the_streamed_reasoning_key_appears(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"reasoning_content": "kimi style"}), finish_event())
        message = stream.result()["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "kimi style")
        self.assertNotIn("reasoning", message)
        self.assertEqual(message["content"], "")

    def test_refusal_deltas_concatenate(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"refusal": "I can't"}),
            delta_event({"refusal": " help with that."}),
            finish_event(),
        )
        message = stream.result()["choices"][0]["message"]
        self.assertEqual(message["refusal"], "I can't help with that.")
        self.assertEqual(message["content"], "")

    def test_non_string_delta_fields_are_ignored(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": None, "reasoning": None, "refusal": None, "tool_calls": None}),
            delta_event({"content": ["not", "a", "string"], "tool_calls": "bad"}),
            delta_event({"content": "ok"}),
            json.dumps({"choices": [{"index": 0, "delta": "not a dict"}]}),
            json.dumps({"choices": ["not a dict", 7]}),
            json.dumps({"choices": "not a list"}),
            finish_event(),
        )
        message = stream.result()["choices"][0]["message"]
        self.assertEqual(message, {"role": "assistant", "content": "ok"})
        self.assertEqual(len(stream.result()["choices"]), 1)

    def test_finish_reason_and_native_finish_reason(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "cut"}),
            json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length",
                            "native_finish_reason": "max_output_tokens",
                        }
                    ]
                }
            ),
        )
        choice = stream.result()["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(choice["native_finish_reason"], "max_output_tokens")
        self.assertTrue(stream.terminal)

    def test_native_finish_reason_is_omitted_when_not_streamed(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "x"}), finish_event("stop"))
        self.assertNotIn("native_finish_reason", stream.result()["choices"][0])

    def test_choice_without_integer_index_defaults_to_zero(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "a"}, index=None),
            json.dumps({"choices": [{"index": True, "delta": {"content": "b"}}]}),
            json.dumps({"choices": [{"index": "1", "delta": {"content": "c"}}]}),
            finish_event(),
        )
        choices = stream.result()["choices"]
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["message"]["content"], "abc")

    def test_every_choice_must_finish_before_terminal(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "second"}, index=1),
            delta_event({"content": "first"}, index=0),
            finish_event("stop", index=0),
        )
        self.assertFalse(stream.terminal)
        stream.feed(finish_event("stop", index=1))
        self.assertTrue(stream.terminal)
        choices = stream.result()["choices"]
        self.assertEqual([choice["index"] for choice in choices], [0, 1])
        self.assertEqual([choice["message"]["content"] for choice in choices], ["first", "second"])


class AccumulatorToolCallTests(unittest.TestCase):
    def test_fragments_merge_by_index(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": ""},
                        }
                    ]
                }
            ),
            delta_event(
                {"tool_calls": [{"index": 0, "id": "", "function": {"name": "", "arguments": '{"q": '}}]}
            ),
            delta_event(
                {"tool_calls": [{"index": 0, "id": None, "function": {"name": None, "arguments": '"x"}'}}]}
            ),
            delta_event(
                {"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "fetch", "arguments": "{}"}}]}
            ),
            finish_event("tool_calls"),
        )
        choice = stream.result()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "")
        self.assertEqual(
            choice["message"]["tool_calls"],
            [
                {"id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
            ],
        )
        for call in choice["message"]["tool_calls"]:
            self.assertNotIn("index", call)
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"q": "x"})

    def test_out_of_order_indexes_are_sorted(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "second", "arguments": "{\"b\""}}]}),
            delta_event({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "first", "arguments": "{\"a\""}}]}),
            delta_event({"tool_calls": [{"index": 1, "function": {"arguments": ": 2}"}}]}),
            delta_event({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
            finish_event("tool_calls"),
        )
        calls = stream.result()["choices"][0]["message"]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual([call["function"]["name"] for call in calls], ["first", "second"])
        self.assertEqual([call["function"]["arguments"] for call in calls], ['{"a": 1}', '{"b": 2}'])

    def test_fragments_without_index_continue_the_latest_or_matching_call(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"tool_calls": [{"id": "call_a", "function": {"name": "lookup", "arguments": "{\"q\""}}]}),
            delta_event({"tool_calls": [{"function": {"arguments": ": 1}"}}]}),
            delta_event({"tool_calls": [{"id": "call_b", "function": {"name": "other", "arguments": "{"}}]}),
            delta_event({"tool_calls": [{"id": "call_a", "function": {"arguments": ""}}]}),
            delta_event({"tool_calls": [{"function": {"arguments": "}"}}]}),
            finish_event("tool_calls"),
        )
        calls = stream.result()["choices"][0]["message"]["tool_calls"]
        self.assertEqual(
            calls,
            [
                {"id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}},
                {"id": "call_b", "type": "function", "function": {"name": "other", "arguments": "{}"}},
            ],
        )

    def test_type_defaults_to_function_and_can_be_streamed(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "f", "arguments": "{}"}}]}),
            delta_event({"tool_calls": [{"index": 1, "id": "c1", "type": "custom", "function": {"name": "g"}}]}),
            finish_event("tool_calls"),
        )
        calls = stream.result()["choices"][0]["message"]["tool_calls"]
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[1]["type"], "custom")
        self.assertEqual(calls[1]["function"], {"name": "g", "arguments": ""})


class AccumulatorUsageTests(unittest.TestCase):
    USAGE = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "cost": 0.000123,
        "prompt_tokens_details": {"cached_tokens": 4},
    }

    def test_trailing_usage_chunk_repeating_finish_reason_is_not_double_counted(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "Hi"}),
            finish_event("stop"),
            finish_event("stop", usage=self.USAGE),
        )
        result = stream.result()
        self.assertTrue(stream.terminal)
        self.assertEqual(len(result["choices"]), 1)
        self.assertEqual(result["choices"][0]["message"], {"role": "assistant", "content": "Hi"})
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"], self.USAGE)
        self.assertEqual(result["usage"]["cost"], 0.000123)
        self.assertEqual(stream.valid_chunks, 3)

    def test_usage_chunk_with_empty_choices_is_kept(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "Hi"}),
            finish_event("stop", usage=None),
            json.dumps({"choices": [], "usage": self.USAGE}),
        )
        result = stream.result()
        self.assertTrue(stream.terminal)
        self.assertEqual(len(result["choices"]), 1)
        self.assertEqual(result["usage"], self.USAGE)

    def test_null_usage_after_real_usage_keeps_the_real_usage(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            finish_event("stop", usage=self.USAGE),
            json.dumps({"choices": [], "usage": None}),
        )
        self.assertEqual(stream.result()["usage"], self.USAGE)

    def test_usage_is_omitted_when_never_streamed(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "x"}), finish_event())
        self.assertNotIn("usage", stream.result())

    def test_usage_on_the_final_choice_is_kept(self) -> None:
        # Kimi reports streamed usage on the last choice rather than on the chunk.
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "Hi"}),
            json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "usage": self.USAGE}]}),
        )
        result = stream.result()
        self.assertTrue(stream.terminal)
        self.assertEqual(result["usage"], self.USAGE)
        self.assertNotIn("usage", result["choices"][0])


class AccumulatorErrorTests(unittest.TestCase):
    UPSTREAM_ERROR = {"code": 502, "message": "Upstream overloaded", "metadata": {"provider_name": "HostB"}}

    def test_error_object_as_first_and_only_event(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        self.assertTrue(stream.feed(json.dumps({"id": "gen-err", "provider": "HostA", "error": self.UPSTREAM_ERROR})))
        self.assertTrue(stream.failed)
        self.assertFalse(stream.terminal)
        self.assertEqual(stream.error, self.UPSTREAM_ERROR)
        self.assertEqual(stream.result()["choices"], [])

    def test_error_after_partial_content_fails_even_with_a_finish_reason(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "par"}),
            finish_event("error", error={"code": "server_error", "message": "mid-stream failure"}),
        )
        self.assertTrue(stream.failed)
        self.assertFalse(stream.terminal)
        choice = stream.result()["choices"][0]
        self.assertEqual(choice["message"]["content"], "par")
        self.assertEqual(choice["finish_reason"], "error")

    def test_error_with_a_normal_finish_reason_is_still_not_terminal(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "x"}), finish_event("stop", error={"message": "late failure"}))
        self.assertTrue(stream.failed)
        self.assertFalse(stream.terminal)

    def test_choice_level_error_is_recorded(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        stream.feed(json.dumps({"choices": [{"index": 0, "delta": {}, "error": {"message": "choice failed"}}]}))
        self.assertTrue(stream.failed)
        self.assertEqual(stream.error, {"message": "choice failed"})

    def test_string_error_is_recorded_and_blank_string_is_ignored(self) -> None:
        blank = ChatCompletionStreamAccumulator()
        feed_all(blank, json.dumps({"error": "   "}), json.dumps({"error": None}), json.dumps({"error": 5}))
        self.assertFalse(blank.failed)
        self.assertIsNone(blank.error)

        stream = ChatCompletionStreamAccumulator()
        stream.feed(json.dumps({"error": "boom"}))
        self.assertTrue(stream.failed)
        self.assertEqual(stream.error, "boom")
        self.assertEqual(stream.error_body()["error"], {"message": "boom"})

    def test_finish_reason_error_without_error_object_fails(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "x"}), finish_event("error"))
        self.assertIsNone(stream.error)
        self.assertTrue(stream.failed)
        self.assertFalse(stream.terminal)
        self.assertEqual(
            stream.error_body()["error"],
            {"message": "The provider ended the stream with finish_reason 'error'"},
        )

    def test_first_error_is_kept(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, json.dumps({"error": {"message": "first"}}), json.dumps({"error": {"message": "second"}}))
        self.assertEqual(stream.error, {"message": "first"})


class AccumulatorFramingEdgeTests(unittest.TestCase):
    def test_malformed_and_non_object_chunks_are_counted(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        results = feed_all(stream, "{not json", "[1, 2]", '"text"', "42", "null")
        self.assertEqual(results, [True] * 5)
        self.assertEqual(stream.malformed_chunks, 5)
        self.assertEqual(stream.valid_chunks, 0)
        self.assertFalse(stream.terminal)
        self.assertFalse(stream.failed)
        self.assertEqual(stream.result()["choices"], [])

    def test_malformed_chunk_keeps_a_finished_stream_incomplete(self) -> None:
        # An unreadable chunk may have carried part of the answer, so it is never accepted.
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "a"}), '{"choices": [', delta_event({"content": "b"}), finish_event())
        self.assertEqual(stream.malformed_chunks, 1)
        self.assertFalse(stream.failed)
        self.assertFalse(stream.terminal)

    def test_native_finish_reason_alone_does_not_finish_a_choice(self) -> None:
        # OpenRouter always normalizes finish_reason; a raw MAX_TOKENS alone would hide an output-limit cut-off.
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"reasoning": "thinking about it..."}),
            json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": None, "native_finish_reason": "MAX_TOKENS"}]}),
            DONE_SENTINEL,
        )
        self.assertFalse(stream.failed)
        self.assertFalse(stream.terminal)

    def test_parallel_tool_calls_reusing_an_index_stay_separate(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        for call_id, query in (("call_a", "a"), ("call_b", "b")):
            feed_all(stream, delta_event({"tool_calls": [{
                "index": -1,
                "id": call_id,
                "type": "function",
                "function": {"name": "wolfram_alpha_query", "arguments": ""},
            }]}))
            feed_all(stream, delta_event({"tool_calls": [{"index": -1, "function": {"arguments": json.dumps({"query": query})}}]}))
        feed_all(stream, finish_event("tool_calls"))
        calls = stream.result()["choices"][0]["message"]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in calls], [{"query": "a"}, {"query": "b"}])

    def test_done_sentinel_stops_feeding(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        self.assertTrue(stream.feed(delta_event({"content": "x"})))
        self.assertFalse(stream.done_seen)
        self.assertFalse(stream.feed(DONE_SENTINEL))
        self.assertTrue(stream.done_seen)

        padded = ChatCompletionStreamAccumulator()
        self.assertFalse(padded.feed("  [DONE]\n"))
        self.assertTrue(padded.done_seen)

    def test_done_without_finish_reason_is_not_terminal(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(stream, delta_event({"content": "unfinished"}), DONE_SENTINEL)
        self.assertTrue(stream.done_seen)
        self.assertFalse(stream.terminal)
        self.assertFalse(stream.failed)

    def test_blank_events_are_ignored(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        self.assertEqual(feed_all(stream, "", "   ", "\n"), [True, True, True])
        self.assertEqual(stream.valid_chunks, 0)
        self.assertEqual(stream.malformed_chunks, 0)

    def test_empty_stream_is_not_terminal(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        self.assertFalse(stream.terminal)
        self.assertFalse(stream.failed)
        self.assertFalse(stream.done_seen)
        self.assertEqual(
            stream.result("vendor/fallback"),
            {"id": "", "object": "chat.completion", "created": None, "model": "vendor/fallback", "choices": []},
        )

        done_only = ChatCompletionStreamAccumulator()
        done_only.feed(DONE_SENTINEL)
        self.assertFalse(done_only.terminal)

    def test_role_only_chunk_is_not_terminal(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        stream.feed(delta_event({"role": "assistant"}))
        self.assertFalse(stream.terminal)
        self.assertEqual(stream.result()["choices"][0]["finish_reason"], None)


class AccumulatorResultShapeTests(unittest.TestCase):
    def test_message_content_is_always_a_string(self) -> None:
        tool_only = ChatCompletionStreamAccumulator()
        feed_all(
            tool_only,
            delta_event({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]}),
            finish_event("tool_calls"),
        )
        reasoning_only = ChatCompletionStreamAccumulator()
        feed_all(reasoning_only, delta_event({"reasoning": "hmm"}), finish_event("length"))
        unfinished = ChatCompletionStreamAccumulator()
        unfinished.feed(delta_event({"role": "assistant"}))
        for stream in (tool_only, reasoning_only, unfinished):
            message = stream.result()["choices"][0]["message"]
            self.assertIsInstance(message["content"], str)
            self.assertEqual(message["content"], "")
            self.assertEqual(message["role"], "assistant")

    def test_model_defaults_and_first_metadata_wins(self) -> None:
        bare = ChatCompletionStreamAccumulator()
        feed_all(bare, delta_event({"content": "x"}), finish_event())
        self.assertEqual(bare.result("vendor/fallback")["model"], "vendor/fallback")
        self.assertEqual(bare.result()["model"], "")
        self.assertNotIn("provider", bare.result())
        self.assertIsNone(bare.result()["created"])

        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "x"}, id="gen-first", model="vendor/first", provider="HostFirst", created=111),
            finish_event(id="gen-second", model="vendor/second", provider="HostSecond", created=222),
        )
        result = stream.result("vendor/fallback")
        self.assertEqual(result["id"], "gen-first")
        self.assertEqual(result["model"], "vendor/first")
        self.assertEqual(result["provider"], "HostFirst")
        self.assertEqual(result["created"], 111)

    def test_non_integer_created_and_non_string_metadata_are_ignored(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "x"}, id=7, model=None, provider={"name": "x"}, created=True),
            finish_event(created="1700000000"),
            finish_event(created=1700000001, id="gen-ok"),
        )
        result = stream.result("vendor/fallback")
        self.assertEqual(result["created"], 1700000001)
        self.assertEqual(result["id"], "gen-ok")
        self.assertEqual(result["model"], "vendor/fallback")
        self.assertNotIn("provider", result)


class AccumulatorErrorBodyTests(unittest.TestCase):
    def test_error_object_with_upstream_metadata(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        stream.feed(
            json.dumps(
                {
                    "id": "gen-9",
                    "provider": "HostA",
                    "error": {"code": 502, "message": "Upstream overloaded", "metadata": {"provider_name": "HostB"}},
                }
            )
        )
        body = stream.error_body()
        self.assertEqual(
            body,
            {
                "error": {"code": 502, "message": "Upstream overloaded", "metadata": {"provider_name": "HostB"}},
                "id": "gen-9",
                "provider": "HostA",
            },
        )
        self.assertEqual(
            _reported_upstream(body),
            {
                "upstream_host": "HostB",
                "upstream_error_code": "502",
                "upstream_error_message": "Upstream overloaded",
                "request_id": "gen-9",
            },
        )

    def test_streamed_provider_is_the_host_when_metadata_is_absent(self) -> None:
        stream = ChatCompletionStreamAccumulator()
        feed_all(
            stream,
            delta_event({"content": "par"}, id="gen-10", provider="HostA"),
            json.dumps({"error": {"code": "server_error", "message": "mid-stream failure"}}),
        )
        reported = _reported_upstream(stream.error_body())
        self.assertEqual(reported["upstream_host"], "HostA")
        self.assertEqual(reported["upstream_error_code"], "server_error")
        self.assertEqual(reported["upstream_error_message"], "mid-stream failure")
        self.assertEqual(reported["request_id"], "gen-10")

    def test_string_error_and_bare_finish_error_bodies(self) -> None:
        string_error = ChatCompletionStreamAccumulator()
        string_error.feed(json.dumps({"error": "boom"}))
        body = string_error.error_body()
        self.assertEqual(body, {"error": {"message": "boom"}, "id": ""})
        reported = _reported_upstream(body)
        self.assertEqual(reported["upstream_error_message"], "boom")
        self.assertEqual(reported["upstream_error_code"], "")
        self.assertEqual(reported["upstream_host"], "")
        self.assertEqual(reported["request_id"], "")

        bare = ChatCompletionStreamAccumulator()
        feed_all(bare, delta_event({"content": "x"}, id="gen-11"), finish_event("error"))
        reported = _reported_upstream(bare.error_body())
        self.assertEqual(reported["upstream_error_code"], "")
        self.assertIn("finish_reason 'error'", reported["upstream_error_message"])
        self.assertEqual(reported["request_id"], "gen-11")


if __name__ == "__main__":
    unittest.main()
