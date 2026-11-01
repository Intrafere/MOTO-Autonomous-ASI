"""Server-sent-event framing and chunk reassembly for streamed Chat Completions.

Provider-neutral: imports only the standard library and httpx so any provider
client can stream without an import cycle. Reassembly reports provider errors as
data instead of raising, so each client keeps its own classification policy.
"""
import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

DONE_SENTINEL = "[DONE]"
# SSE lines end only at CR, LF or CRLF. str.splitlines (and httpx's aiter_lines)
# also break at U+0085/U+2028/U+2029, which JSON leaves unescaped in strings.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


class _SseFramer:
    """Join ``data:`` lines into events; events end at a blank line, ``:`` lines are comments."""

    def __init__(self) -> None:
        self._data_lines: List[str] = []

    def push(self, line: str) -> Optional[str]:
        if not line:
            return self.flush()
        if line.startswith("data:"):
            self._data_lines.append(line[5:].lstrip())
        return None

    def flush(self) -> Optional[str]:
        if not self._data_lines:
            return None
        event = "\n".join(self._data_lines)
        self._data_lines = []
        return event


def _complete_lines(buffer: str, final: bool) -> Tuple[List[str], str]:
    """Split off every terminated line; a trailing CR waits for a possible LF unless final."""
    lines: List[str] = []
    start = 0
    for match in _LINE_BREAK.finditer(buffer):
        if not final and match.group() == "\r" and match.end() == len(buffer):
            break
        lines.append(buffer[start:match.start()])
        start = match.end()
    remainder = buffer[start:]
    if final and remainder:
        lines.append(remainder)
        remainder = ""
    return lines, remainder


def split_sse_data(raw_body: str) -> List[str]:
    """Return each event's joined ``data:`` payload from a fully buffered SSE body."""
    framer = _SseFramer()
    lines, _ = _complete_lines(raw_body, final=True)
    events = [event for line in lines if (event := framer.push(line)) is not None]
    trailing = framer.flush()
    if trailing is not None:
        events.append(trailing)
    return events


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield each event's joined ``data:`` payload as a streamed response arrives."""
    framer = _SseFramer()
    pending: List[str] = []
    async for text in response.aiter_text():
        # Hold pieces of an unterminated line so a long line is joined and scanned once.
        if not _LINE_BREAK.search(text) and not (pending and pending[-1].endswith("\r")):
            pending.append(text)
            continue
        lines, remainder = _complete_lines("".join(pending) + text, final=False)
        pending = [remainder] if remainder else []
        for line in lines:
            event = framer.push(line)
            if event is not None:
                yield event
    lines, _ = _complete_lines("".join(pending), final=True)
    for line in lines:
        event = framer.push(line)
        if event is not None:
            yield event
    trailing = framer.flush()
    if trailing is not None:
        yield trailing


def is_event_stream(response: httpx.Response) -> bool:
    return "text/event-stream" in response.headers.get("content-type", "").lower()


class ChatCompletionStreamAccumulator:
    """Rebuild the non-streaming Chat Completions response from streamed chunks.

    Content, reasoning (kept under the key the provider streamed), refusals and
    tool-call fragments merge per choice index. A chunk's ``error`` object is kept
    as data. Usage may arrive on the chunk or, as Kimi sends it, on a choice. A
    trailing usage chunk that repeats ``finish_reason`` is an accounting frame, not
    a second completion. An unreadable chunk could have carried part of the answer,
    so a stream with one is never complete.
    """

    def __init__(self) -> None:
        self._choices: Dict[int, Dict[str, Any]] = {}
        self.response_id = ""
        self.model = ""
        self.provider = ""
        self.created: Optional[int] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.error: Any = None
        self.done_seen = False
        self.valid_chunks = 0
        self.malformed_chunks = 0

    def feed(self, event_data: str) -> bool:
        """Absorb one event payload; return False once the ``[DONE]`` sentinel arrives."""
        data = event_data.strip()
        if not data:
            return True
        if data == DONE_SENTINEL:
            self.done_seen = True
            return False
        try:
            chunk = json.loads(data)
        except ValueError:
            self.malformed_chunks += 1
            return True
        if not isinstance(chunk, dict):
            self.malformed_chunks += 1
            return True
        self.valid_chunks += 1
        self._absorb(chunk)
        return True

    def _absorb(self, chunk: Dict[str, Any]) -> None:
        for attribute, key in (("response_id", "id"), ("model", "model"), ("provider", "provider")):
            value = chunk.get(key)
            if not getattr(self, attribute) and isinstance(value, str):
                setattr(self, attribute, value)
        created = chunk.get("created")
        if self.created is None and isinstance(created, int) and not isinstance(created, bool):
            self.created = created
        self._record_error(chunk.get("error"))
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        choices = chunk.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if isinstance(choice, dict):
                self._absorb_choice(choice)

    def _record_error(self, error: Any) -> None:
        if self.error is None and (isinstance(error, dict) or (isinstance(error, str) and error.strip())):
            self.error = error

    def _absorb_choice(self, choice: Dict[str, Any]) -> None:
        index = choice.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            index = 0
        state = self._choices.setdefault(
            index,
            {"content": [], "reasoning": {}, "refusal": [], "tool_calls": [], "finish_reason": None, "native_finish_reason": None},
        )
        self._record_error(choice.get("error"))
        if isinstance(choice.get("usage"), dict):
            self.usage = choice["usage"]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            if isinstance(delta.get("content"), str):
                state["content"].append(delta["content"])
            for key in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    state["reasoning"].setdefault(key, []).append(delta[key])
            if isinstance(delta.get("refusal"), str):
                state["refusal"].append(delta["refusal"])
            tool_calls = delta.get("tool_calls")
            for call in tool_calls if isinstance(tool_calls, list) else []:
                if isinstance(call, dict):
                    self._merge_tool_call(state["tool_calls"], call)
        for key in ("finish_reason", "native_finish_reason"):
            if choice.get(key):
                state[key] = str(choice[key])

    @staticmethod
    def _merge_tool_call(slots: List[Dict[str, Any]], call: Dict[str, Any]) -> None:
        index = call.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            index = None
        call_id = str(call.get("id") or "")
        slot = None
        if index is not None:
            slot = next((existing for existing in reversed(slots) if existing["index"] == index), None)
            # A new id at a reused index is a separate parallel call.
            if slot is not None and call_id and slot["id"] and slot["id"] != call_id:
                slot = None
        elif call_id:
            slot = next((existing for existing in reversed(slots) if existing["id"] == call_id), None)
        elif slots:
            slot = slots[-1]
        if slot is None:
            slot = {
                "index": len(slots) if index is None else index,
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
            slots.append(slot)
        if call_id:
            slot["id"] = call_id
        if call.get("type"):
            slot["type"] = str(call["type"])
        function = call.get("function")
        if isinstance(function, dict):
            if function.get("name"):
                slot["function"]["name"] = str(function["name"])
            if isinstance(function.get("arguments"), str):
                slot["function"]["arguments"] += function["arguments"]

    @property
    def failed(self) -> bool:
        """True when the provider reported an error, including a bare ``finish_reason: error``."""
        return self.error is not None or any(
            state["finish_reason"] == "error" for state in self._choices.values()
        )

    @property
    def terminal(self) -> bool:
        """True when every choice finished, no error was reported and every chunk was readable."""
        return (
            bool(self._choices)
            and not self.failed
            and not self.malformed_chunks
            and all(state["finish_reason"] for state in self._choices.values())
        )

    def result(self, default_model: str = "") -> Dict[str, Any]:
        choices: List[Dict[str, Any]] = []
        for index in sorted(self._choices):
            state = self._choices[index]
            message: Dict[str, Any] = {"role": "assistant", "content": "".join(state["content"])}
            for key, parts in state["reasoning"].items():
                message[key] = "".join(parts)
            if state["refusal"]:
                message["refusal"] = "".join(state["refusal"])
            if state["tool_calls"]:
                message["tool_calls"] = [
                    {key: value for key, value in slot.items() if key != "index"}
                    for slot in sorted(state["tool_calls"], key=lambda slot: slot["index"])
                ]
            choice: Dict[str, Any] = {"index": index, "message": message, "finish_reason": state["finish_reason"]}
            if state["native_finish_reason"]:
                choice["native_finish_reason"] = state["native_finish_reason"]
            choices.append(choice)
        data: Dict[str, Any] = {
            "id": self.response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model or default_model,
            "choices": choices,
        }
        if self.provider:
            data["provider"] = self.provider
        if self.usage is not None:
            data["usage"] = self.usage
        return data

    def error_body(self) -> Dict[str, Any]:
        """The streamed failure in OpenRouter's non-streaming error-body shape."""
        error = self.error
        if isinstance(error, str):
            error = {"message": error}
        elif not isinstance(error, dict):
            error = {"message": "The provider ended the stream with finish_reason 'error'"}
        body: Dict[str, Any] = {"error": error, "id": self.response_id}
        if self.provider:
            body["provider"] = self.provider
        return body
