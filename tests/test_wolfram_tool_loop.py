import json
import unittest

from backend.compiler.agents import high_context_submitter as submitter_module
from backend.compiler.agents.high_context_submitter import HighContextSubmitter
from backend.shared import wolfram_alpha_client as wolfram_module


class FakeWolframClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def query(self, query: str) -> str:
        self.queries.append(query)
        return f"result for {query}"


def _tool_call(call_id: str, query: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "wolfram_alpha_query",
            "arguments": json.dumps({"query": query, "purpose": "test"}),
        },
    }


class WolframToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_executes_query_and_returns_final_json(self) -> None:
        submitter = HighContextSubmitter(model_name="test-model", user_prompt="Write.")
        fake_client = FakeWolframClient()
        broadcasts = []

        async def broadcaster(event: str, data: dict) -> None:
            broadcasts.append((event, data))

        submitter.websocket_broadcaster = broadcaster
        responses = [
            {"choices": [{"message": {"content": "", "tool_calls": [_tool_call("call-1", "2+2")]}}]},
            {"choices": [{"message": {"content": '{"needs_construction": false}'}}]},
        ]
        calls = []

        async def fake_generate_completion(**kwargs):
            calls.append(kwargs)
            return responses.pop(0)

        original_available = submitter_module._wolfram_tool_available
        original_get_client = wolfram_module.get_wolfram_client
        original_generate = submitter_module.api_client_manager.generate_completion
        try:
            submitter_module._wolfram_tool_available = lambda: True
            wolfram_module.get_wolfram_client = lambda: fake_client
            submitter_module.api_client_manager.generate_completion = fake_generate_completion

            content, wolfram_calls, _message = await submitter._generate_completion_with_wolfram_tool(
                task_id="task-1",
                initial_prompt="prompt",
            )
        finally:
            submitter_module._wolfram_tool_available = original_available
            wolfram_module.get_wolfram_client = original_get_client
            submitter_module.api_client_manager.generate_completion = original_generate

        self.assertEqual(content, '{"needs_construction": false}')
        self.assertEqual(fake_client.queries, ["2+2"])
        self.assertEqual(wolfram_calls[0]["query"], "2+2")
        self.assertIsNotNone(calls[0]["tools"])
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(broadcasts[0][0], "compiler_wolfram_call")

    async def test_single_turn_multiple_tool_calls_cannot_exceed_budget(self) -> None:
        submitter = HighContextSubmitter(model_name="test-model", user_prompt="Write.")
        fake_client = FakeWolframClient()
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call("call-1", "1+1"),
                                _tool_call("call-2", "2+2"),
                                _tool_call("call-3", "3+3"),
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": '{"needs_construction": false}'}}]},
        ]

        async def fake_generate_completion(**kwargs):
            return responses.pop(0)

        original_available = submitter_module._wolfram_tool_available
        original_get_client = wolfram_module.get_wolfram_client
        original_generate = submitter_module.api_client_manager.generate_completion
        original_budget = submitter_module.WOLFRAM_MAX_CALLS_PER_SUBMISSION
        try:
            submitter_module._wolfram_tool_available = lambda: True
            wolfram_module.get_wolfram_client = lambda: fake_client
            submitter_module.api_client_manager.generate_completion = fake_generate_completion
            submitter_module.WOLFRAM_MAX_CALLS_PER_SUBMISSION = 2

            _content, wolfram_calls, _message = await submitter._generate_completion_with_wolfram_tool(
                task_id="task-2",
                initial_prompt="prompt",
            )
        finally:
            submitter_module._wolfram_tool_available = original_available
            wolfram_module.get_wolfram_client = original_get_client
            submitter_module.api_client_manager.generate_completion = original_generate
            submitter_module.WOLFRAM_MAX_CALLS_PER_SUBMISSION = original_budget

        self.assertEqual(fake_client.queries, ["1+1", "2+2"])
        self.assertEqual(len(wolfram_calls), 2)


if __name__ == "__main__":
    unittest.main()
