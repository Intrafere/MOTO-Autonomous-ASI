"""Repeated model-output contract failures in Autonomous loops back off, never spin.

The first contract failure of a step is re-asked at once; later consecutive
failures wait 1, 2, 4, ... (at most 30) seconds through the shared stop-aware
emit-and-wait seam. Re-asks are never capped, and every reported failure is its
own Live Activity item.
"""
from __future__ import annotations

import asyncio
import importlib
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import backend.autonomous.core.autonomous_coordinator  # noqa: F401
from backend.autonomous.agents import output_contract
from backend.autonomous.agents.final_answer.certainty_assessor import CertaintyAssessor
from backend.autonomous.agents.paper_title_selector import PaperTitleSelectorAgent
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.shared.activity import normalize_activity_payload
from backend.shared.api_client_manager import api_client_manager
from backend.shared.model_output_contract import (
    ModelOutputContractError,
    contract_retry_delay_seconds,
    wait_after_repeated_contract_failure,
)
from backend.shared.models import CertaintyAssessment, TopicSelectionSubmission

coordinator_module = importlib.import_module("backend.autonomous.core.autonomous_coordinator")
certainty_module = importlib.import_module(
    "backend.autonomous.agents.final_answer.certainty_assessor"
)


def _contract_error():
    return ModelOutputContractError("x", stage="syntax")


class _BackoffTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.waits = []

        async def capture(event, data=None):
            self.events.append((event, data or {}))

        async def emit(payload, **kwargs):
            self.waits.append((payload, kwargs))
            return True

        for patcher in (
            mock.patch.object(api_client_manager, "_broadcast", new=capture),
            mock.patch.object(api_client_manager, "emit_and_wait_for_retry", new=emit),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def delays(self):
        return [payload["retry_after_seconds"] for payload, _ in self.waits]

    def contract_events(self):
        return [data for event, data in self.events if event == "model_output_contract_failed"]


class SharedBackoffTests(_BackoffTestCase):
    def test_delay_schedule_is_immediate_then_doubling_to_thirty(self):
        self.assertEqual(
            [contract_retry_delay_seconds(n) for n in range(1, 10)],
            [0, 1, 2, 4, 8, 16, 30, 30, 30],
        )

    async def test_wait_uses_the_seam_and_is_stop_aware(self):
        stopped = False
        await wait_after_repeated_contract_failure(
            "autonomous_topic_selector", 3, should_stop=lambda: stopped
        )
        (payload, kwargs), = self.waits
        self.assertEqual(payload["retry_after_seconds"], 2)
        self.assertEqual(payload["retry_attempt"], 3)
        self.assertEqual(payload["retry_layer"], "workflow")
        self.assertEqual(payload["reason"], "model_output_contract_retry")
        self.assertIn("not a provider error", payload["message"])
        self.assertIn("not a rejection", payload["message"])
        self.assertEqual(kwargs["role_id"], "autonomous_topic_selector")
        stopped = True
        self.assertTrue(kwargs["should_stop"]())

    async def test_no_wait_once_stopped(self):
        await wait_after_repeated_contract_failure(
            "autonomous_topic_selector", 5, should_stop=lambda: True
        )
        self.assertEqual(self.waits, [])

    async def test_pacer_counts_consecutive_failures_and_resets(self):
        pacer = output_contract.ContractFailurePacer()
        for _ in range(3):
            await pacer.after_failure("role_a")
        pacer.reset()
        await pacer.after_failure("role_a")
        await pacer.after_failure("role_a")
        self.assertEqual(self.delays(), [1, 2, 1])


class ContractFailureActivityIdentityTests(_BackoffTestCase):
    async def test_repeated_failures_of_one_role_are_distinct_items(self):
        published = [
            await output_contract.report_model_output_contract_failure(
                "autonomous_topic_selector", _contract_error(), model="gemma"
            )
            for _ in range(2)
        ]
        ids = {item["event_id"] for item in published}
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(event_id.startswith("model_output_contract_failed:") for event_id in ids))
        activity_ids = {
            normalize_activity_payload("model_output_contract_failed", item)["activity_id"]
            for item in published
        }
        self.assertEqual(len(activity_ids), 2)


class TopicSelectionBackoffTests(_BackoffTestCase):
    async def test_repeated_selector_contract_failures_back_off_without_a_cap(self):
        coordinator = AutonomousCoordinator()
        submission = TopicSelectionSubmission(action="new_topic", topic_prompt="Q", reasoning="R")
        failures = 7
        coordinator._topic_selector = SimpleNamespace(
            role_id="autonomous_topic_selector",
            model_id="gemma",
            select_topic=AsyncMock(side_effect=[_contract_error()] * failures + [submission]),
            handle_rejection=AsyncMock(),
        )
        coordinator._topic_validator = SimpleNamespace(
            role_id="autonomous_topic_validator",
            model_id="gemma",
            validate=AsyncMock(return_value=SimpleNamespace(decision="accept", reasoning="ok")),
        )
        coordinator._execute_topic_selection = AsyncMock(return_value="topic_010")
        rag = SimpleNamespace(
            get_all_brainstorms_summary=AsyncMock(return_value=[]),
            get_all_papers_summary=AsyncMock(return_value=[]),
        )
        with mock.patch.object(coordinator_module, "autonomous_rag_manager", rag):
            topic_id = await coordinator._topic_selection_loop()
        self.assertEqual(topic_id, "topic_010")
        self.assertEqual(coordinator._topic_selector.select_topic.await_count, failures + 1)
        self.assertEqual(self.delays(), [1, 2, 4, 8, 16, 30])
        self.assertEqual(len(self.contract_events()), failures)
        _, kwargs = self.waits[0]
        self.assertFalse(kwargs["should_stop"]())
        coordinator._stop_event.set()
        self.assertTrue(kwargs["should_stop"]())

    async def test_repeated_validator_contract_failures_back_off(self):
        coordinator = AutonomousCoordinator()
        submission = TopicSelectionSubmission(action="new_topic", topic_prompt="Q", reasoning="R")
        coordinator._topic_selector = SimpleNamespace(
            role_id="autonomous_topic_selector",
            model_id="gemma",
            select_topic=AsyncMock(return_value=submission),
            handle_rejection=AsyncMock(),
        )
        coordinator._topic_validator = SimpleNamespace(
            role_id="autonomous_topic_validator",
            model_id="gemma",
            validate=AsyncMock(
                side_effect=[_contract_error()] * 3
                + [SimpleNamespace(decision="accept", reasoning="ok")]
            ),
        )
        coordinator._execute_topic_selection = AsyncMock(return_value="topic_011")
        rag = SimpleNamespace(
            get_all_brainstorms_summary=AsyncMock(return_value=[]),
            get_all_papers_summary=AsyncMock(return_value=[]),
        )
        with mock.patch.object(coordinator_module, "autonomous_rag_manager", rag):
            topic_id = await coordinator._topic_selection_loop()
        self.assertEqual(topic_id, "topic_011")
        coordinator._topic_selector.select_topic.assert_awaited_once()
        self.assertEqual(self.delays(), [1, 2])
        self.assertEqual(
            {payload["role_id"] for payload, _ in self.waits}, {"autonomous_topic_validator"}
        )


class ContinuationBackoffTests(_BackoffTestCase):
    async def test_repeated_continuation_contract_failures_back_off(self):
        rag = SimpleNamespace(
            get_brainstorm_summary=AsyncMock(return_value="summary"),
            get_all_brainstorms_summary=AsyncMock(return_value=[]),
            get_all_papers_summary=AsyncMock(return_value=[]),
        )
        brainstorms = SimpleNamespace(
            get_metadata=AsyncMock(return_value=SimpleNamespace(topic_prompt="Topic"))
        )
        metadata = SimpleNamespace(get_papers_by_brainstorm=AsyncMock(return_value=[]))
        coordinator = AutonomousCoordinator()
        coordinator._current_topic_id = "topic_001"
        callbacks = []
        coordinator._topic_selector = SimpleNamespace(
            model_id="gemma",
            context_window=32000,
            max_output_tokens=1000,
            task_sequence=0,
            task_tracking_callback=lambda state, task_id: callbacks.append(state),
        )
        coordinator._topic_validator = SimpleNamespace(
            role_id="autonomous_topic_validator",
            model_id="gemma",
            validate=AsyncMock(return_value=SimpleNamespace(decision="accept", reasoning="ok")),
        )
        bad = {"choices": [{"message": {"content": '{"reasoning": "no decision"}'}}]}
        good = {
            "choices": [
                {"message": {"content": '{"decision": "move_on", "reasoning": "Done."}'}}
            ]
        }
        # Each failed ask makes two calls (the reply and its one bounded retry).
        completions = AsyncMock(side_effect=[bad] * 6 + [good])
        with mock.patch.object(coordinator_module, "autonomous_rag_manager", rag), \
                mock.patch.object(coordinator_module, "brainstorm_memory", brainstorms), \
                mock.patch.object(coordinator_module, "research_metadata", metadata), \
                mock.patch.object(api_client_manager, "generate_completion", new=completions):
            decision = await coordinator._brainstorm_continuation_decision()
        self.assertEqual(decision, "move_on")
        self.assertEqual(self.delays(), [1, 2])
        # The task is marked completed before each backoff wait, never left open.
        self.assertEqual(callbacks, ["started", "completed"] * 4)


class TitleSelectionBackoffTests(_BackoffTestCase):
    async def test_repeated_validator_contract_failures_back_off_stop_aware(self):
        selector = PaperTitleSelectorAgent("sub", "val", context_window=32000, max_output_tokens=1000)
        selection = SimpleNamespace(paper_title="A Title", reasoning="R")
        selector._generate_title = AsyncMock(return_value=selection)
        selector._validate_title = AsyncMock(side_effect=[_contract_error()] * 4 + [(True, "")])
        stop_event = asyncio.Event()
        title = await selector.select_title("prompt", "topic", "summary", [], stop_event=stop_event)
        self.assertEqual(title, "A Title")
        selector._generate_title.assert_awaited_once()
        self.assertEqual(self.delays(), [1, 2, 4])
        _, kwargs = self.waits[0]
        stop_event.set()
        self.assertTrue(kwargs["should_stop"]())

    async def test_generation_contract_failures_back_off(self):
        selector = PaperTitleSelectorAgent("sub", "val", context_window=32000, max_output_tokens=1000)
        selection = SimpleNamespace(paper_title="A Title", reasoning="R")
        selector._generate_title = AsyncMock(side_effect=[_contract_error()] * 3 + [selection])
        selector._validate_title = AsyncMock(return_value=(True, ""))
        title = await selector.select_title("prompt", "topic", "summary", [])
        self.assertEqual(title, "A Title")
        self.assertEqual(self.delays(), [1, 2])
        self.assertEqual(len(self.contract_events()), 3)


class Tier3BackoffTests(_BackoffTestCase):
    async def test_certainty_contract_failures_back_off_on_the_bound_stop_state(self):
        memory = SimpleNamespace(
            get_rejections=AsyncMock(return_value=[]),
            add_rejection=AsyncMock(),
            save_certainty_assessment=AsyncMock(),
        )
        assessor = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        stop_event = asyncio.Event()
        assessor.should_stop = stop_event.is_set
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        assessor._request_paper_expansion = AsyncMock(return_value=[])
        assessor._generate_assessment = AsyncMock(
            side_effect=[_contract_error()] * 3 + [assessment]
        )
        assessor._validate_assessment = AsyncMock(return_value=(True, "Sound."))
        with mock.patch.object(certainty_module, "final_answer_memory", memory):
            result = await assessor.assess_certainty("prompt", [{"paper_id": "p1"}])
        self.assertIs(result, assessment)
        self.assertEqual(self.delays(), [1, 2])
        _, kwargs = self.waits[0]
        stop_event.set()
        self.assertTrue(kwargs["should_stop"]())

    def test_unbound_tier3_agents_never_report_stopped(self):
        assessor = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        self.assertFalse(assessor.should_stop())


if __name__ == "__main__":
    unittest.main()
