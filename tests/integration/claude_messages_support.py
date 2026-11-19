"""Test helpers for providers that use the Claude Messages API wire.

The Anthropic SDK has its own HTTP client, so a test handler written against
``httpx`` requests and responses is bridged into the SDK's transport. Everything
below the SDK call (request building, SSE parsing, typed errors) stays real.
"""
import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import httpx2


def bridged_messages_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx2.AsyncClient:
    """Return an SDK HTTP client that answers every request with ``handler``."""

    async def bridge(request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        response = handler(
            httpx.Request(request.method, str(request.url), headers=dict(request.headers), content=body)
        )
        headers = {
            name: value for name, value in response.headers.items()
            if name.lower() not in {"content-length", "content-encoding"}
        }
        return httpx2.Response(response.status_code, headers=headers, content=response.content)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(bridge))


def _event(name: str, data: Dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def messages_sse(
    text: str = "OK",
    *,
    model: str = "claude-opus-5-5",
    stop_reason: Optional[str] = "end_turn",
    tool_use: Optional[Dict[str, Any]] = None,
    thinking_signature: str = "",
    usage: Optional[Dict[str, int]] = None,
    stop_details: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build a Messages API event stream for one assistant message."""
    usage = usage or {"input_tokens": 5, "output_tokens": 2}
    start_usage = {key: value for key, value in usage.items() if key != "output_tokens"}
    events: List[str] = [_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {**start_usage, "output_tokens": 0},
        },
    })]
    index = 0

    def block(start: Dict[str, Any], deltas: List[Dict[str, Any]]) -> None:
        nonlocal index
        events.append(_event("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": start,
        }))
        for delta in deltas:
            events.append(_event("content_block_delta", {
                "type": "content_block_delta", "index": index, "delta": delta,
            }))
        events.append(_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1

    if thinking_signature:
        block(
            {"type": "thinking", "thinking": "", "signature": ""},
            [{"type": "signature_delta", "signature": thinking_signature}],
        )
    if text:
        block({"type": "text", "text": ""}, [{"type": "text_delta", "text": text}])
    if tool_use:
        block(
            {"type": "tool_use", "id": tool_use["id"], "name": tool_use["name"], "input": {}},
            [{"type": "input_json_delta", "partial_json": json.dumps(tool_use.get("input") or {})}],
        )
    if stop_reason is not None:
        delta: Dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
        if stop_details is not None:
            delta["stop_details"] = stop_details
        events.append(_event("message_delta", {
            "type": "message_delta", "delta": delta,
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        }))
        events.append(_event("message_stop", {"type": "message_stop"}))
    return "".join(events).encode("utf-8")


def messages_response(text: str = "OK", **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=messages_sse(text, **kwargs))


def messages_stream_error(error_type: str, message: str) -> httpx.Response:
    """Return a stream that starts normally (HTTP 200) and then reports an error event."""
    body = messages_sse("partial", stop_reason=None).decode("utf-8") + _event(
        "error", {"type": "error", "error": {"type": error_type, "message": message}},
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode("utf-8"))


def messages_error(status: int, error_type: str, message: str, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        json={"type": "error", "error": {"type": error_type, "message": message}},
    )
