"""Model-output contract handling for Autonomous strategy and Tier 3 roles.

Contract failures (malformed or missing-field replies) get one bounded retry and
then surface as operational activity: never a semantic rejection, a provider
retry wait, or a silently defaulted decision.
"""
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

# The coordinator imports the agents in dependency order.
import backend.autonomous.core.autonomous_coordinator  # noqa: F401
from backend.autonomous.agents import output_contract
from backend.autonomous.agents.completion_reviewer import CompletionReviewerAgent
from backend.autonomous.agents.final_answer.answer_format_selector import AnswerFormatSelector
from backend.autonomous.agents.final_answer.certainty_assessor import CertaintyAssessor
from backend.autonomous.agents.final_answer.volume_organizer import (
    VolumeOrganizer,
    _parse_organization,
)
from backend.autonomous.agents.paper_title_selector import PaperTitleSelectorAgent
from backend.autonomous.agents.reference_selector import ReferenceSelectorAgent
from backend.autonomous.agents.topic_validator import TopicValidatorAgent
from backend.autonomous.core.autonomous_coordinator import (
    PROOF_FRAMING_UNRESOLVED_MESSAGE,
    AutonomousCoordinator,
)
from backend.autonomous.prompts.proof_prompts import PROOF_FRAMING_CONTEXT
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.model_error_utils import is_transient_model_call_error
from backend.shared.model_output_contract import ModelOutputContractError
from backend.shared.models import (
    AnswerFormatSelection,
    CertaintyAssessment,
    CompletionReviewResult,
    TopicSelectionSubmission,
)

coordinator_module = importlib.import_module("backend.autonomous.core.autonomous_coordinator")
completion_module = importlib.import_module("backend.autonomous.agents.completion_reviewer")
certainty_module = importlib.import_module(
    "backend.autonomous.agents.final_answer.certainty_assessor"
)
format_module = importlib.import_module(
    "backend.autonomous.agents.final_answer.answer_format_selector"
)
volume_module = importlib.import_module("backend.autonomous.agents.final_answer.volume_organizer")


def _provider_error():
    return RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="role_x",
        model="model-x",
        reason="upstream_timeout",
        message="upstream gateway timeout",
    )


def _reply(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {"choices": [{"message": {"content": content}}]}


class _ScriptedCompletions:
    """Async generate_completion stand-in returning scripted replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return _reply(reply)


class _ContractTestCase(unittest.IsolatedAsyncioTestCase):
    """Patches the shared singleton so every module sees scripted completions."""

    def setUp(self):
        self.events = []

        async def capture(event, data=None):
            self.events.append((event, data or {}))

        patches = [
            mock.patch.object(api_client_manager, "_broadcast", new=capture),
            mock.patch.object(
                api_client_manager, "prewarm_assistant_memory_context", new=AsyncMock()
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def script(self, *replies):
        completions = _ScriptedCompletions(*replies)
        patcher = mock.patch.object(api_client_manager, "generate_completion", new=completions)
        patcher.start()
        self.addCleanup(patcher.stop)
        return completions

    def contract_events(self):
        return [data for event, data in self.events if event == "model_output_contract_failed"]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


class FieldCheckTests(unittest.TestCase):
    def test_string_false_is_not_a_boolean(self):
        with self.assertRaises(ModelOutputContractError) as ctx:
            output_contract.require_bool({"flag": "false"}, "flag", role="r")
        self.assertEqual(ctx.exception.stage, "invalid_value")
        self.assertEqual(ctx.exception.invalid_fields, ("flag",))

    def test_missing_choice_names_the_field(self):
        with self.assertRaises(ModelOutputContractError) as ctx:
            output_contract.require_choice({}, "decision", ("accept", "reject"), role="r")
        self.assertEqual(ctx.exception.stage, "missing_key")
        self.assertEqual(ctx.exception.missing_keys, ("decision",))

    def test_syntax_failure_never_quotes_raw_output(self):
        secret = "PRIVATE-MODEL-TEXT-123"
        with self.assertRaises(ModelOutputContractError) as ctx:
            output_contract.parse_contract_reply(
                f'{{"decision": {secret}', lambda data: data, role="r"
            )
        self.assertEqual(ctx.exception.stage, "syntax")
        self.assertNotIn(secret, str(ctx.exception))

    def test_parser_value_errors_are_wrapped_without_their_text(self):
        def parser(_data):
            raise ValueError("model said PRIVATE-TEXT")

        with self.assertRaises(ModelOutputContractError) as ctx:
            output_contract.parse_contract_reply('{"a": 1}', parser, role="r")
        self.assertNotIn("PRIVATE-TEXT", str(ctx.exception))

    def test_contract_errors_are_never_provider_failures(self):
        error = ModelOutputContractError(
            "The response is missing a required field",
            stage="missing_key",
            missing_keys=("timeout",),
        )
        self.assertFalse(api_client_manager.is_provider_failure(error))
        self.assertFalse(is_transient_model_call_error(error))


class ContractRetryTests(_ContractTestCase):
    async def _parse(self, first_content):
        return await output_contract.parse_with_contract_retry(
            first_content,
            lambda data: output_contract.require_choice(
                output_contract.json_object(data, role="role_x"),
                "decision",
                ("accept", "reject"),
                role="role_x",
            ),
            prompt="ORIGINAL PROMPT",
            task_id="agg_val_001",
            role_id="role_x",
            model="model-x",
            context_window=32000,
            max_output_tokens=1000,
            response_noun="validation decision",
            response_shape='{"decision": "accept | reject"}',
        )

    async def test_valid_first_reply_makes_no_retry(self):
        completions = self.script()
        self.assertEqual(await self._parse('{"decision": "accept"}'), "accept")
        self.assertEqual(completions.calls, [])

    async def test_one_retry_names_the_failed_field(self):
        completions = self.script({"decision": "reject"})
        self.assertEqual(await self._parse('{"decison": "accept"}'), "reject")
        self.assertEqual(len(completions.calls), 1)
        call = completions.calls[0]
        self.assertEqual(call["task_id"], "agg_val_001_retry")
        self.assertEqual(call["role_id"], "role_x")
        self.assertEqual(call["messages"][0]["content"], "ORIGINAL PROMPT")
        retry_text = call["messages"][-1]["content"]
        self.assertIn("Your previous validation decision response could not be used.", retry_text)
        self.assertIn("'decision'", retry_text)
        self.assertIn("Respond with ONLY the JSON object", retry_text)

    async def test_second_failure_raises_typed_error(self):
        self.script("still not json")
        with self.assertRaises(ModelOutputContractError) as ctx:
            await self._parse("not json")
        self.assertEqual(ctx.exception.stage, "syntax")
        self.assertIsInstance(ctx.exception.__cause__, ModelOutputContractError)

    async def test_provider_failure_on_retry_propagates(self):
        self.script(_provider_error())
        with self.assertRaises(RetryableProviderError):
            await self._parse("not json")

    async def test_contract_failure_activity_is_operational(self):
        error = ModelOutputContractError(
            "The response is missing a required field",
            stage="missing_key",
            missing_keys=("decision",),
        )
        published = await output_contract.report_model_output_contract_failure(
            "autonomous_topic_validator",
            error,
            task_id="agg_val_002",
            model="gemma",
            outcome="It will be validated again.",
        )
        self.assertIsNotNone(published)
        (event, data), = self.events
        self.assertEqual(event, "model_output_contract_failed")
        self.assertEqual(data["activity_classification"], "operational")
        self.assertEqual(data["activity_outcome"], "contract_failed")
        self.assertEqual(data["missing_keys"], ["decision"])
        self.assertIn("after one retry (decision)", data["message"])
        self.assertIn("It will be validated again.", data["message"])


# ---------------------------------------------------------------------------
# Proof framing gate (D8)
# ---------------------------------------------------------------------------


class _FramingMetadata:
    def __init__(self):
        self.framing_states = []
        self.workflow_states = []

    async def set_proof_framing_state(self, **state):
        self.framing_states.append(dict(state))

    async def save_workflow_state(self, state):
        self.workflow_states.append(dict(state))

    async def begin_lifecycle(self, _run_id):
        return 1

    async def get_workflow_state(self):
        return dict(self.workflow_states[-1]) if self.workflow_states else {}

    def has_interrupted_workflow(self):
        return bool(self.workflow_states)


class ProofFramingGateTests(_ContractTestCase):
    def setUp(self):
        super().setUp()
        self.metadata = _FramingMetadata()
        patcher = mock.patch.object(coordinator_module, "research_metadata", self.metadata)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _coordinator(self):
        coordinator = AutonomousCoordinator()
        coordinator._allow_mathematical_proofs = True
        coordinator._base_user_research_prompt = "Bound the chromatic number."
        coordinator._submitter_configs = [
            SimpleNamespace(
                submitter_id=1, model_id="gemma", context_window=8192, max_output_tokens=700
            )
        ]
        coordinator._save_workflow_state = AsyncMock()
        coordinator.set_broadcast_callback(self._capture_coordinator)
        return coordinator

    async def _capture_coordinator(self, event, data):
        self.events.append((event, data))

    async def test_provider_failure_propagates_without_persisting_a_decision(self):
        self.script(_provider_error())
        coordinator = self._coordinator()
        with self.assertRaises(RetryableProviderError):
            await coordinator._run_proof_framing_gate()
        self.assertEqual(self.metadata.framing_states, [])
        self.assertTrue(coordinator._proof_framing_gate_pending)

    async def test_string_false_is_a_contract_failure_not_true(self):
        completions = self.script(
            '{"is_proof_amenable": "false", "reasoning": "x"}',
            {"is_proof_amenable": False, "reasoning": "Empirical."},
        )
        coordinator = self._coordinator()
        coordinator._proof_framing_unresolved = True
        await coordinator._run_proof_framing_gate()
        self.assertEqual(len(completions.calls), 2)
        self.assertFalse(coordinator._proof_framing_active)
        self.assertFalse(coordinator._proof_framing_unresolved)
        self.assertEqual(self.metadata.framing_states[-1]["reasoning"], "Empirical.")

    async def test_unresolved_gate_defaults_framing_on_and_is_visible(self):
        self.script("not json", "still not json")
        coordinator = self._coordinator()
        await coordinator._run_proof_framing_gate()

        self.assertTrue(coordinator._proof_framing_active)
        self.assertEqual(coordinator._proof_framing_context, PROOF_FRAMING_CONTEXT)
        self.assertTrue(coordinator._proof_framing_unresolved)
        self.assertFalse(coordinator._proof_framing_gate_pending)
        persisted = self.metadata.framing_states[-1]
        self.assertTrue(persisted["active"])
        self.assertEqual(persisted["reasoning"], PROOF_FRAMING_UNRESOLVED_MESSAGE)
        decided = [data for event, data in self.events if event == "proof_framing_decided"]
        self.assertTrue(decided[-1]["unresolved"])
        self.assertEqual(decided[-1]["message"], PROOF_FRAMING_UNRESOLVED_MESSAGE)
        self.assertEqual(len(self.contract_events()), 1)

    async def test_unresolved_marker_is_persisted_and_reruns_gate_on_resume(self):
        coordinator = self._coordinator()
        coordinator._proof_framing_unresolved = True
        del coordinator._save_workflow_state
        await coordinator._save_workflow_state()
        self.assertTrue(self.metadata.workflow_states[-1]["proof_framing_unresolved"])

        self.assertTrue(await self._resume_with_marker(True))

    async def test_resume_state_restores_unresolved_marker(self):
        for persisted_marker, allow_proofs, expected in (
            (True, True, True),
            ("true", True, False),
            (True, False, False),
        ):
            with self.subTest(marker=persisted_marker, allow_proofs=allow_proofs):
                coordinator = AutonomousCoordinator()
                coordinator._allow_mathematical_proofs = allow_proofs
                coordinator._user_research_prompt = "Research prompt"
                state = {
                    "current_tier": "tier1_aggregation",
                    "proof_framing_active": True,
                    "proof_framing_unresolved": persisted_marker,
                }
                metadata = mock.Mock()
                metadata.has_interrupted_workflow.return_value = True
                metadata.get_workflow_state = AsyncMock(return_value=state)
                metadata.get_base_user_prompt = AsyncMock(return_value="Research prompt")
                metadata.set_proof_framing_state = AsyncMock()
                with mock.patch.object(
                    coordinator, "_normalize_resume_state", new=AsyncMock(return_value=state)
                ), mock.patch.object(coordinator_module, "research_metadata", metadata):
                    await coordinator._check_resume_state()
                self.assertIs(coordinator._proof_framing_unresolved, expected)

    async def test_resume_without_marker_skips_gate(self):
        self.assertFalse(await self._resume_with_marker(False))

    async def _resume_with_marker(self, unresolved):
        """Return whether start() schedules the gate for a resumed run."""
        resumed = AutonomousCoordinator()
        resumed._allow_mathematical_proofs = True
        resumed._proof_framing_unresolved = unresolved
        resumed.refresh_workflow_predictions = AsyncMock()
        resumed._broadcast_stopped_once = AsyncMock()
        resumed._run_proof_framing_gate = AsyncMock()

        async def resume_point_then_stop():
            resumed._stop_event.set()
            return {"current_tier": "tier1_aggregation"}

        resumed._get_resume_point = AsyncMock(side_effect=resume_point_then_stop)
        await resumed.start()
        resumed._run_proof_framing_gate.assert_not_awaited()
        return resumed._proof_framing_gate_pending


# ---------------------------------------------------------------------------
# Completion self-validation (rule-defined fallback, no durable feedback)
# ---------------------------------------------------------------------------


class CompletionReviewerTests(_ContractTestCase):
    def setUp(self):
        super().setUp()
        self.logs = SimpleNamespace(add_completion_feedback=AsyncMock())
        patcher = mock.patch.object(completion_module, "autonomous_rejection_logs", self.logs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _reviewer(self):
        reviewer = CompletionReviewerAgent("gemma", context_window=32000, max_output_tokens=1000)
        reviewer._prepare_brainstorm_context = AsyncMock(return_value=("db", False))
        reviewer._generate_assessment = AsyncMock(
            return_value=CompletionReviewResult(
                decision="write_paper", reasoning="Ready.", suggested_additions=""
            )
        )
        return reviewer

    async def test_unusable_self_validation_continues_without_feedback(self):
        self.script("not json", '{"validated": "yes"}')
        result, validated = await self._reviewer().review_completion(
            "prompt", "topic_001", "topic", "db", 9
        )
        self.assertFalse(validated)
        self.assertEqual(result.decision, "continue_brainstorm")
        self.assertIn("rule-defined fallback", result.reasoning)
        self.logs.add_completion_feedback.assert_not_awaited()
        self.assertEqual(len(self.contract_events()), 1)

    async def test_semantic_self_validation_failure_still_records_feedback(self):
        self.script({"validated": False, "reasoning": "A concrete gap remains."})
        result, validated = await self._reviewer().review_completion(
            "prompt", "topic_001", "topic", "db", 9
        )
        self.assertFalse(validated)
        self.assertEqual(result.decision, "continue_brainstorm")
        self.logs.add_completion_feedback.assert_awaited_once()

    async def test_assessment_missing_decision_is_not_a_decision(self):
        reviewer = CompletionReviewerAgent("gemma", context_window=32000, max_output_tokens=1000)
        self.logs.get_completion_feedback = AsyncMock(return_value=[])
        self.logs.render_completion_feedback = lambda entries: ""
        self.script({"reasoning": "no decision"}, {"reasoning": "still none"})
        result = await reviewer._generate_assessment("prompt", "topic_001", "topic", "db", 9, False)
        self.assertIsNone(result)
        self.assertEqual(self.contract_events()[0]["missing_keys"], ["decision"])


# ---------------------------------------------------------------------------
# Validators raise typed errors instead of fabricating rejections
# ---------------------------------------------------------------------------


class ValidatorContractTests(_ContractTestCase):
    async def test_topic_validator_contract_failure_is_not_a_rejection(self):
        self.script({"reasoning": "missing decision"}, "not json")
        validator = TopicValidatorAgent("gemma", context_window=32000, max_output_tokens=1000)
        with self.assertRaises(ModelOutputContractError):
            await validator.validate(
                submission=TopicSelectionSubmission(
                    action="new_topic", topic_prompt="Q", reasoning="R"
                ),
                user_research_prompt="prompt",
                brainstorms_summary=[],
                papers_summary=[],
                override_prompt="VALIDATE THIS",
            )

    async def test_topic_validator_still_returns_semantic_rejections(self):
        self.script({"decision": "reject", "reasoning": "Duplicate topic."})
        validator = TopicValidatorAgent("gemma", context_window=32000, max_output_tokens=1000)
        result = await validator.validate(
            submission=TopicSelectionSubmission(action="new_topic", topic_prompt="Q", reasoning="R"),
            user_research_prompt="prompt",
            brainstorms_summary=[],
            papers_summary=[],
            override_prompt="VALIDATE THIS",
        )
        self.assertEqual((result.decision, result.reasoning), ("reject", "Duplicate topic."))

    async def test_tier3_validators_no_longer_default_to_reject(self):
        certainty = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        volume = VolumeOrganizer("sub", "val", context_window=32000, max_output_tokens=1000)
        answer_format = AnswerFormatSelector(
            "sub", "val", context_window=32000, max_output_tokens=1000
        )
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        organization = _parse_organization(_valid_volume(), role="r")
        cases = (
            lambda: certainty._validate_assessment("prompt", [], assessment),
            lambda: volume._validate_organization("prompt", [], organization),
            lambda: answer_format._validate_selection(
                "prompt",
                assessment,
                [],
                AnswerFormatSelection(answer_format="short_form", reasoning="R"),
            ),
        )
        for make_call in cases:
            with self.subTest():
                self.script({"reasoning": "no decision"}, {"verdict": "accept"})
                with self.assertRaises(ModelOutputContractError) as ctx:
                    await make_call()
                self.assertEqual(ctx.exception.missing_keys, ("decision",))

    async def test_certainty_loop_revalidates_instead_of_recording_rejection(self):
        memory = SimpleNamespace(
            get_rejections=AsyncMock(return_value=[]),
            add_rejection=AsyncMock(),
            save_certainty_assessment=AsyncMock(),
        )
        assessor = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        assessor._request_paper_expansion = AsyncMock(return_value=[])
        assessor._generate_assessment = AsyncMock(return_value=assessment)
        assessor._validate_assessment = AsyncMock(
            side_effect=[
                ModelOutputContractError("x", stage="missing_key", missing_keys=("decision",)),
                (True, "Sound."),
            ]
        )
        with mock.patch.object(certainty_module, "final_answer_memory", memory):
            result = await assessor.assess_certainty("prompt", [{"paper_id": "p1"}])
        self.assertIs(result, assessment)
        assessor._generate_assessment.assert_awaited_once()
        self.assertEqual(assessor._validate_assessment.await_count, 2)
        memory.add_rejection.assert_not_awaited()
        self.assertEqual(len(self.contract_events()), 1)

    async def test_format_loop_revalidates_instead_of_recording_rejection(self):
        memory = SimpleNamespace(
            get_rejections=AsyncMock(return_value=[]),
            add_rejection=AsyncMock(),
            set_answer_format=AsyncMock(),
        )
        selector = AnswerFormatSelector("sub", "val", context_window=32000, max_output_tokens=1000)
        selection = AnswerFormatSelection(answer_format="long_form", reasoning="R")
        selector._generate_selection = AsyncMock(return_value=selection)
        selector._validate_selection = AsyncMock(
            side_effect=[ModelOutputContractError("x", stage="syntax"), (True, "Fine.")]
        )
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        with mock.patch.object(format_module, "final_answer_memory", memory):
            result = await selector.select_format("prompt", assessment, [{"paper_id": "p1"}])
        self.assertIs(result, selection)
        selector._generate_selection.assert_awaited_once()
        memory.add_rejection.assert_not_awaited()

    async def test_volume_loop_revalidates_instead_of_recording_rejection(self):
        memory = SimpleNamespace(
            get_rejections=AsyncMock(return_value=[]),
            add_rejection=AsyncMock(),
            save_volume_organization=AsyncMock(),
        )
        organizer = VolumeOrganizer("sub", "val", context_window=32000, max_output_tokens=1000)
        organization = _parse_organization(_valid_volume(), role="r")
        organizer._generate_organization = AsyncMock(return_value=organization)
        organizer._validate_organization = AsyncMock(
            side_effect=[ModelOutputContractError("x", stage="syntax"), (True, "Fine.")]
        )
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        with mock.patch.object(volume_module, "final_answer_memory", memory):
            result = await organizer.organize_volume("prompt", assessment, [{"paper_id": "p1"}])
        self.assertIs(result, organization)
        organizer._generate_organization.assert_awaited_once()
        memory.add_rejection.assert_not_awaited()

    async def test_tier3_contract_failures_never_use_up_fixed_tries(self):
        """More contract failures than MAX_RETRIES/MAX_ITERATIONS still reach a usable reply."""
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        selection = AnswerFormatSelection(answer_format="long_form", reasoning="R")
        organization = _parse_organization(_valid_volume(), role="r")

        certainty = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        certainty._request_paper_expansion = AsyncMock(return_value=[])
        certainty._generate_assessment = AsyncMock(return_value=assessment)
        selector = AnswerFormatSelector("sub", "val", context_window=32000, max_output_tokens=1000)
        selector._generate_selection = AsyncMock(return_value=selection)
        organizer = VolumeOrganizer("sub", "val", context_window=32000, max_output_tokens=1000)
        organizer._generate_organization = AsyncMock(return_value=organization)

        cases = (
            (certainty, "_validate_assessment", certainty_module, certainty.MAX_RETRIES,
             lambda: certainty.assess_certainty("prompt", [{"paper_id": "p1"}]), assessment),
            (selector, "_validate_selection", format_module, selector.MAX_RETRIES,
             lambda: selector.select_format("prompt", assessment, [{"paper_id": "p1"}]), selection),
            (organizer, "_validate_organization", volume_module, organizer.MAX_ITERATIONS,
             lambda: organizer.organize_volume("prompt", assessment, [{"paper_id": "p1"}]),
             organization),
        )
        for agent, validate_name, module, limit, run, expected in cases:
            with self.subTest(agent=type(agent).__name__):
                failures = [ModelOutputContractError("x", stage="syntax")] * (limit + 3)
                setattr(agent, validate_name, AsyncMock(side_effect=[*failures, (True, "Fine.")]))
                memory = SimpleNamespace(
                    get_rejections=AsyncMock(return_value=[]),
                    add_rejection=AsyncMock(),
                    save_certainty_assessment=AsyncMock(),
                    set_answer_format=AsyncMock(),
                    save_volume_organization=AsyncMock(),
                )
                with mock.patch.object(module, "final_answer_memory", memory), mock.patch.object(
                    output_contract, "wait_after_repeated_contract_failure", AsyncMock()
                ):
                    result = await run()
                self.assertIs(result, expected)
                self.assertEqual(getattr(agent, validate_name).await_count, limit + 4)
                memory.add_rejection.assert_not_awaited()

    async def test_tier3_contract_failure_loop_ends_on_stop(self):
        assessment = CertaintyAssessment(
            certainty_level="partial_answer", known_certainties_summary="S", reasoning="R"
        )
        certainty = CertaintyAssessor("sub", "val", context_window=32000, max_output_tokens=1000)
        certainty._request_paper_expansion = AsyncMock(return_value=[])
        certainty._generate_assessment = AsyncMock(return_value=assessment)
        certainty._validate_assessment = AsyncMock(
            side_effect=ModelOutputContractError("x", stage="syntax")
        )
        stopped = {"value": False}

        async def stop_during_wait(*_args, **_kwargs):
            stopped["value"] = True

        certainty.should_stop = lambda: stopped["value"]
        memory = SimpleNamespace(
            get_rejections=AsyncMock(return_value=[]),
            add_rejection=AsyncMock(),
            save_certainty_assessment=AsyncMock(),
        )
        with mock.patch.object(certainty_module, "final_answer_memory", memory), mock.patch.object(
            output_contract, "wait_after_repeated_contract_failure", stop_during_wait
        ):
            result = await certainty.assess_certainty("prompt", [{"paper_id": "p1"}])
        self.assertIsNone(result)
        self.assertEqual(certainty._validate_assessment.await_count, 1)
        memory.add_rejection.assert_not_awaited()


def _valid_volume():
    return {
        "volume_title": "Volume",
        "outline_complete": True,
        "reasoning": "R",
        "chapters": [
            {"chapter_type": "introduction", "paper_id": None, "title": "Intro", "order": 1},
            {"chapter_type": "existing_paper", "paper_id": "paper_1", "title": "Body", "order": 2},
            {"chapter_type": "conclusion", "paper_id": None, "title": "End", "order": 3},
        ],
    }


class VolumeChapterContractTests(unittest.TestCase):
    def test_bad_chapters_are_named_individually(self):
        data = _valid_volume()
        data["chapters"][1].pop("paper_id")
        data["chapters"][2]["order"] = "three"
        with self.assertRaises(ModelOutputContractError) as ctx:
            _parse_organization(data, role="r")
        message = str(ctx.exception)
        self.assertIn("2 of 3 chapter entries were invalid", message)
        self.assertIn("chapters[1]: missing required field 'paper_id'", message)
        self.assertIn("chapters[2]: invalid field 'order'", message)


# ---------------------------------------------------------------------------
# Selection loops
# ---------------------------------------------------------------------------


class TitleSelectionTests(_ContractTestCase):
    async def test_validator_contract_failure_revalidates_same_title(self):
        selector = PaperTitleSelectorAgent("sub", "val", context_window=32000, max_output_tokens=1000)
        selection = SimpleNamespace(paper_title="A Title", reasoning="R")
        selector._generate_title = AsyncMock(return_value=selection)
        selector._validate_title = AsyncMock(
            side_effect=[ModelOutputContractError("x", stage="syntax"), (True, "")]
        )
        with mock.patch.object(api_client_manager, "emit_and_wait_for_retry", new=AsyncMock()) as wait:
            title = await selector.select_title("prompt", "topic", "summary", [])
        self.assertEqual(title, "A Title")
        selector._generate_title.assert_awaited_once()
        self.assertEqual(selector._validate_title.await_count, 2)
        wait.assert_not_awaited()
        self.assertEqual(self.contract_events()[0]["role_id"], "autonomous_paper_title_validator")

    async def test_generation_contract_failure_is_not_a_provider_wait(self):
        selector = PaperTitleSelectorAgent("sub", "val", context_window=32000, max_output_tokens=1000)
        selection = SimpleNamespace(paper_title="A Title", reasoning="R")
        selector._generate_title = AsyncMock(
            side_effect=[ModelOutputContractError("x", stage="syntax"), selection]
        )
        selector._validate_title = AsyncMock(return_value=(True, ""))
        with mock.patch.object(api_client_manager, "emit_and_wait_for_retry", new=AsyncMock()) as wait:
            title = await selector.select_title("prompt", "topic", "summary", [])
        self.assertEqual(title, "A Title")
        wait.assert_not_awaited()


class ReferenceSelectionTests(_ContractTestCase):
    async def test_final_selection_contract_failure_is_not_an_empty_selection(self):
        self.script({"reasoning": "forgot the list"}, {"selected_papers": "paper_1"})
        selector = ReferenceSelectorAgent("gemma", context_window=32000, max_output_tokens=1000)
        with self.assertRaises(ModelOutputContractError) as ctx:
            await selector._make_final_selection(
                "prompt",
                "topic",
                "summary",
                [{"paper_id": "paper_1", "title": "T", "content": "body", "word_count": 1}],
                mode="initial",
                max_papers=3,
            )
        self.assertEqual(ctx.exception.invalid_fields, ("selected_papers",))

    async def test_coordinator_reports_and_adds_no_references(self):
        coordinator = AutonomousCoordinator()
        coordinator._reference_selector = SimpleNamespace(
            role_id="autonomous_reference_selector",
            model_id="gemma",
            select_references=AsyncMock(
                side_effect=ModelOutputContractError("x", stage="syntax")
            ),
        )
        self.assertEqual(await coordinator._select_references_or_report(mode="initial"), [])
        self.assertEqual(len(self.contract_events()), 1)


class TopicSelectionLoopTests(_ContractTestCase):
    async def test_validator_contract_failure_revalidates_without_rejection(self):
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
                side_effect=[
                    ModelOutputContractError("x", stage="syntax"),
                    SimpleNamespace(decision="accept", reasoning="Good."),
                ]
            ),
        )
        coordinator._execute_topic_selection = AsyncMock(return_value="topic_009")
        rag = SimpleNamespace(
            get_all_brainstorms_summary=AsyncMock(return_value=[]),
            get_all_papers_summary=AsyncMock(return_value=[]),
        )
        with mock.patch.object(coordinator_module, "autonomous_rag_manager", rag):
            topic_id = await coordinator._topic_selection_loop()
        self.assertEqual(topic_id, "topic_009")
        coordinator._topic_selector.select_topic.assert_awaited_once()
        self.assertEqual(coordinator._topic_validator.validate.await_count, 2)
        coordinator._topic_selector.handle_rejection.assert_not_awaited()


class ContinuationDecisionTests(_ContractTestCase):
    def _coordinator(self):
        coordinator = AutonomousCoordinator()
        coordinator._current_topic_id = "topic_001"
        coordinator._topic_selector = SimpleNamespace(
            model_id="gemma",
            context_window=32000,
            max_output_tokens=1000,
            task_sequence=0,
            task_tracking_callback=None,
        )
        return coordinator

    def _patch_context(self):
        rag = SimpleNamespace(
            get_brainstorm_summary=AsyncMock(return_value="summary"),
            get_all_brainstorms_summary=AsyncMock(return_value=[]),
            get_all_papers_summary=AsyncMock(return_value=[]),
        )
        brainstorms = SimpleNamespace(
            get_metadata=AsyncMock(return_value=SimpleNamespace(topic_prompt="Topic"))
        )
        metadata = SimpleNamespace(get_papers_by_brainstorm=AsyncMock(return_value=[]))
        for name, value in (
            ("autonomous_rag_manager", rag),
            ("brainstorm_memory", brainstorms),
            ("research_metadata", metadata),
        ):
            patcher = mock.patch.object(coordinator_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_missing_decision_is_asked_again_not_defaulted_to_move_on(self):
        self._patch_context()
        completions = self.script(
            {"reasoning": "no decision"},
            {"reasoning": "still none"},
            {"decision": "write_another_paper", "reasoning": "More to say."},
        )
        coordinator = self._coordinator()
        coordinator._topic_validator = SimpleNamespace(
            role_id="autonomous_topic_validator",
            model_id="gemma",
            validate=AsyncMock(return_value=SimpleNamespace(decision="accept", reasoning="ok")),
        )
        with mock.patch.object(api_client_manager, "emit_and_wait_for_retry", new=AsyncMock()) as wait:
            decision = await coordinator._brainstorm_continuation_decision()
        self.assertEqual(decision, "write_another_paper")
        self.assertEqual(len(completions.calls), 3)
        wait.assert_not_awaited()
        self.assertEqual(self.contract_events()[0]["missing_keys"], ["decision"])

    async def test_validator_contract_failure_revalidates_same_decision(self):
        self._patch_context()
        completions = self.script({"decision": "move_on", "reasoning": "Done."})
        coordinator = self._coordinator()
        coordinator._topic_validator = SimpleNamespace(
            role_id="autonomous_topic_validator",
            model_id="gemma",
            validate=AsyncMock(
                side_effect=[
                    ModelOutputContractError("x", stage="syntax"),
                    SimpleNamespace(decision="accept", reasoning="ok"),
                ]
            ),
        )
        decision = await coordinator._brainstorm_continuation_decision()
        self.assertEqual(decision, "move_on")
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(coordinator._topic_validator.validate.await_count, 2)


class BrainstormReferenceSelectionTests(_ContractTestCase):
    async def test_missing_key_is_a_visible_contract_failure(self):
        completions = self.script({"reasoning": "none"}, {"reasoning": "none again"})
        coordinator = AutonomousCoordinator()
        coordinator._current_topic_id = "topic_002"
        coordinator._submitter_configs = [
            SimpleNamespace(model_id="gemma", context_window=32000, max_output_tokens=1000)
        ]
        coordinator.set_broadcast_callback(self._capture)
        rag = SimpleNamespace(
            get_all_brainstorms_summary=AsyncMock(
                return_value=[
                    {"topic_id": "topic_001", "status": "complete", "submission_count": 4}
                ]
            )
        )
        brainstorms = SimpleNamespace(
            get_metadata=AsyncMock(return_value=SimpleNamespace(topic_prompt="Topic"))
        )
        with mock.patch.object(coordinator_module, "autonomous_rag_manager", rag), \
                mock.patch.object(coordinator_module, "brainstorm_memory", brainstorms):
            selected = await coordinator._pre_brainstorm_reference_brainstorm_selection()
        self.assertEqual(selected, [])
        self.assertEqual(len(completions.calls), 2)
        self.assertIn("'selected_brainstorms'", completions.calls[1]["messages"][0]["content"])
        failed = [data for event, data in self.events if event == "brainstorm_reference_selection_failed"]
        self.assertEqual(failed[-1]["reason"], "model_output_contract_failed")
        self.assertEqual(len(self.contract_events()), 1)

    async def _capture(self, event, data):
        self.events.append((event, data))


if __name__ == "__main__":
    unittest.main()
