import asyncio
import unittest
import tempfile
from types import SimpleNamespace

from backend.leanoj.core import leanoj_context as leanoj_context_module
from backend.leanoj.core import leanoj_coordinator as leanoj_module
from backend.leanoj.core.leanoj_context import LeanOJContextAllocation, leanoj_context_manager
from backend.leanoj.core.leanoj_coordinator import LeanOJConfigurationError, LeanOJCoordinator
from backend.leanoj.prompts import (
    LEANOJ_FORMALIZATION_GUARDRAILS,
    build_final_solution_review_prompt,
    build_final_solver_prompt,
)
from backend.shared.boost_manager import BoostManager
from backend.shared.config import system_config
from backend.shared.models import DocumentChunk, LeanOJRoleConfig, LeanOJStartRequest, ProofRecord
from backend.shared.proof_search.assistant_models import AssistantProofPack, AssistantProofSupport


def _role() -> LeanOJRoleConfig:
    return LeanOJRoleConfig(
        model_id="test-model",
        context_window=8192,
        max_output_tokens=1024,
    )


def _request() -> LeanOJStartRequest:
    return LeanOJStartRequest(
        user_prompt="Prove one equals one.",
        lean_template="import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
        topic_generator=_role(),
        topic_validator=_role(),
        brainstorm_submitters=[_role()],
        brainstorm_validator=_role(),
        path_decider=_role(),
        final_solver=_role(),
    )


class LeanOJCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def _initialized_coordinator(self) -> LeanOJCoordinator:
        coordinator = LeanOJCoordinator()
        await coordinator.initialize(_request())

        async def noop(*_args, **_kwargs):
            return None

        coordinator._persist_state = noop  # type: ignore[method-assign]
        coordinator._broadcast = noop  # type: ignore[method-assign]

        async def fake_register(*_args, **_kwargs):
            return ProofRecord(
                proof_id="proof_leanoj_test",
                theorem_statement="LeanOJ theorem",
                source_type="leanoj_final",
                source_id=coordinator.get_state().session_id,
                lean_code="import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
                novel=True,
                novelty_tier="mathematical_discovery",
                novelty_reasoning="test novelty",
            )

        coordinator._register_verified_leanoj_proof = fake_register  # type: ignore[method-assign]

        async def fake_review(*_args, **_kwargs):
            return True, "Final review accepted.", "Lean 4 accepted with no diagnostics."

        coordinator._review_final_solution_completion = fake_review  # type: ignore[method-assign]
        return coordinator

    def test_final_solver_prompt_has_no_phase_transition_contract(self) -> None:
        prompt = build_final_solver_prompt(
            "Prove one equals one.",
            "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
            "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
            {"version": 1},
            accepted_ideas=[],
            verified_subproofs=[],
            partial_proofs=[],
            failed_feedback=[
                {
                    "request": "final Proof Solver proof cycle",
                    "error_summary": "failed 30 times and requested need_more_brainstorming",
                }
            ],
            final_attempts=[
                {
                    "request": "final attempt",
                    "error_summary": "prior final attempt failed",
                }
            ],
            context_blocks={},
        )

        self.assertNotIn("stuck_needs_brainstorm", prompt)
        self.assertNotIn("need_more_brainstorming", prompt)
        self.assertNotIn("failed 30", prompt)
        self.assertNotIn("brainstorm", prompt.lower())
        self.assertIn("You must choose exactly one action: edit_proof.", prompt)
        self.assertIn(LEANOJ_FORMALIZATION_GUARDRAILS, prompt)
        self.assertIn("truncated natural subtraction", prompt)

    def test_final_solver_prompt_includes_proof_search_context_block(self) -> None:
        prompt = build_final_solver_prompt(
            "Prove one equals one.",
            "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
            "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
            {"version": 1},
            accepted_ideas=[],
            verified_subproofs=[],
            partial_proofs=[],
            failed_feedback=[],
            final_attempts=[],
            context_blocks={
                "proof_search_context": (
                    "Result 1\n"
                    "Source: syntheticlib4 stable\n"
                    "Theorem: searched_helper\n"
                    "Lean code hash: code_hash"
                )
            },
        )

        self.assertIn("SYNTHETIC / LOCAL VERIFIED PROOF SEARCH RESULTS", prompt)
        self.assertIn("searched_helper", prompt)

    async def test_final_solver_proof_search_context_uses_metadata_only_results(self) -> None:
        request = _request()
        coordinator = await self._initialized_coordinator()
        captured = {}

        class FakeAssistantCoordinator:
            def submit_target(self, snapshot):
                captured["snapshot"] = snapshot
                return "target_hash"

            def get_latest_pack(self, target_hash=None):
                captured["target_hash"] = target_hash
                return AssistantProofPack(
                    workflow_mode="leanoj",
                    target_kind="final_solver",
                    target_hash="target_hash",
                    query_summary="LeanOJ final solver",
                    results=[
                        AssistantProofSupport(
                            search_id="syntheticlib4:sl4_mock_fp_001",
                            corpus="syntheticlib4",
                            corpus_scope="stable",
                            source_kind="verified_proof",
                            proof_id="sl4_mock_fp_001",
                            fingerprint="sl4_mock_fp_001",
                            theorem_name="SyntheticLib4.Test.helper",
                            theorem_statement="theorem helper : True",
                            proof_description="A helper proof pattern.",
                            imports=["Mathlib"],
                            dependency_names=["True.intro"],
                            theorem_statement_hash="stmt_hash",
                            lean_code_hash="code_hash",
                            canonical_uri="syntheticlib4://stable/sl4_mock_fp_001",
                            lean_code="theorem helper : True := by\n  trivial",
                        )
                    ],
                )

        original_assistant = leanoj_module.assistant_proof_search_coordinator
        try:
            leanoj_module.assistant_proof_search_coordinator = FakeAssistantCoordinator()
            coordinator._read_master_proof = (  # type: ignore[method-assign]
                lambda: asyncio.sleep(0, result="import Mathlib\n\nexample : True := by\n  sorry")
            )
            context = await coordinator._build_final_solver_proof_search_context(
                request=request,
                task_request="Edit the master proof.",
            )
        finally:
            leanoj_module.assistant_proof_search_coordinator = original_assistant

        self.assertEqual(captured["target_hash"], "target_hash")
        self.assertEqual(captured["snapshot"].workflow_mode, "leanoj")
        self.assertEqual(captured["snapshot"].target_kind, "final_solver")
        self.assertIn("SyntheticLib4.Test.helper", context)
        self.assertIn("syntheticlib4://stable/sl4_mock_fp_001", context)
        self.assertNotIn("theorem helper : True := by", context)

    def test_final_review_prompt_requires_semantic_cross_check(self) -> None:
        prompt = build_final_solution_review_prompt(
            "Solve the informal olympiad problem.",
            "def answer (n : Nat) : Nat := sorry\n\ntheorem solution : True := by\n  sorry",
            "def answer (n : Nat) : Nat := 0\n\ntheorem solution : True := by\n  trivial",
            "Lean should accept.",
            "Lean 4 accepted with no diagnostics.",
        )

        self.assertIn(LEANOJ_FORMALIZATION_GUARDRAILS, prompt)
        self.assertIn("Lean acceptance is necessary but not sufficient", prompt)
        self.assertIn("does not automatically prove the user's informal problem statement", prompt)

    async def test_final_loop_retries_until_lean_verifies(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir

        responses = [
            {"lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  simp", "reasoning": "first try"},
            {"lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  rfl", "reasoning": "fix"},
        ]

        async def fake_call_json(_config, task_prefix, *_args, **_kwargs):
            if task_prefix == "leanoj_master_proof_edit_val":
                raise AssertionError("Tiny placeholder replacement should not require shortening validation")
            if task_prefix == "leanoj_final_review":
                return {"solved": True, "reasoning": "final answer complete"}
            return responses.pop(0)

        class FakeLean:
            def __init__(self) -> None:
                self.calls = 0

            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(success=False, error_output="unsolved goals")
                return SimpleNamespace(success=True, error_output="")

        fake_lean = FakeLean()
        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: fake_lean  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "verified")
                self.assertEqual(coordinator.get_state().final_attempt_count, 2)
                self.assertIn("rfl", coordinator.get_state().final_solution)
                self.assertEqual(coordinator.get_state().final_proof_id, "proof_leanoj_test")
                self.assertEqual(coordinator.get_state().final_novelty_tier, "mathematical_discovery")
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_path_decision_replaces_removed_proof_storm_with_brainstorming(self) -> None:
        request = _request()
        coordinator = await self._initialized_coordinator()
        coordinator._accepted_ideas.append("Useful accepted idea.")

        async def fake_context_blocks(*_args, **_kwargs):
            return {}

        async def fake_call_json(*_args, **_kwargs):
            return {"path": "need_proof_storm", "reasoning": "legacy path"}

        async def fake_validate(*_args, **_kwargs):
            return True, ""

        coordinator._build_context_blocks = fake_context_blocks  # type: ignore[method-assign]
        coordinator._call_json = fake_call_json  # type: ignore[method-assign]
        coordinator._validate_path_decision = fake_validate  # type: ignore[method-assign]

        decision = await coordinator._path_decision_phase(request)

        self.assertEqual(decision, "need_more_brainstorming")
        self.assertNotIn("need_proof_storm", leanoj_module._LEANOJ_PATH_OPTIONS)

    async def test_recursive_brainstorm_starts_proof_memory_without_topic_prepass(self) -> None:
        request = _request().model_copy(
            update={
                "brainstorm_submitters": [
                    _role().model_copy(update={"model_id": "submitter-1"}),
                    _role().model_copy(update={"model_id": "submitter-2"}),
                    _role().model_copy(update={"model_id": "submitter-3"}),
                ],
            }
        )
        coordinator = LeanOJCoordinator()
        await coordinator.initialize(request)

        async def noop(*_args, **_kwargs):
            return None

        brainstorm_calls: list[dict[str, object]] = []
        async def fake_brainstorm_until_path_check(*_args, **_kwargs):
            brainstorm_calls.append(dict(_kwargs))
            return None

        coordinator._persist_state = noop  # type: ignore[method-assign]
        coordinator._broadcast = noop  # type: ignore[method-assign]
        coordinator._brainstorm_until_path_check = fake_brainstorm_until_path_check  # type: ignore[method-assign]

        await coordinator._recursive_brainstorm_phase(request)

        self.assertEqual(len(brainstorm_calls), 1)
        self.assertEqual(brainstorm_calls[0]["phase_key"], "recursive_brainstorm")
        self.assertNotIn("_".join(["recursive", "topics"]), coordinator.get_status())

    async def test_topic_dequeue_respects_remaining_capacity_without_dropping(self) -> None:
        coordinator = LeanOJCoordinator()
        topic_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        await topic_queue.put((1, "topic one"))
        await topic_queue.put((2, "topic two"))
        await topic_queue.put((3, "topic three"))

        batch = await coordinator._dequeue_topic_batch(topic_queue, max_count=1)  # type: ignore[attr-defined]

        self.assertEqual(batch, [(1, "topic one")])
        self.assertEqual(topic_queue.qsize(), 2)

    async def test_brainstorm_queue_global_pause_uses_shared_threshold(self) -> None:
        old_global_threshold = system_config.queue_overflow_threshold
        old_submitter_threshold = system_config.per_submitter_queue_threshold
        try:
            system_config.queue_overflow_threshold = 3
            system_config.per_submitter_queue_threshold = 2
            queue = leanoj_module._LeanOJBrainstormSubmissionQueue(submitter_count=3)  # type: ignore[attr-defined]

            await queue.put((1, "submission one", {}))
            await queue.put((2, "submission two", {}))
            await queue.put((3, "submission three", {}))

            self.assertTrue(queue.should_pause_submitter(1))
            self.assertTrue(queue.should_pause_submitter(2))
            self.assertTrue(queue.should_pause_submitter(3))
            transitions = queue.refresh_pause_transitions()
            self.assertTrue(transitions["global_paused"])
            self.assertTrue(transitions["global_changed"])
        finally:
            system_config.queue_overflow_threshold = old_global_threshold
            system_config.per_submitter_queue_threshold = old_submitter_threshold

    async def test_brainstorm_queue_pauses_only_overrepresented_submitter(self) -> None:
        old_global_threshold = system_config.queue_overflow_threshold
        old_submitter_threshold = system_config.per_submitter_queue_threshold
        try:
            system_config.queue_overflow_threshold = 10
            system_config.per_submitter_queue_threshold = 2
            queue = leanoj_module._LeanOJBrainstormSubmissionQueue(submitter_count=3)  # type: ignore[attr-defined]

            await queue.put((1, "submission one", {}))
            await queue.put((1, "submission two", {}))
            await queue.put((1, "submission three", {}))

            self.assertTrue(queue.should_pause_submitter(1))
            self.assertFalse(queue.should_pause_submitter(2))
            self.assertFalse(queue.should_pause_submitter(3))
            transitions = queue.refresh_pause_transitions()
            self.assertFalse(transitions["global_paused"])
            self.assertEqual(transitions["submitters_paused"], {1})
        finally:
            system_config.queue_overflow_threshold = old_global_threshold
            system_config.per_submitter_queue_threshold = old_submitter_threshold

    async def test_brainstorm_queue_skips_per_submitter_pause_for_single_submitter(self) -> None:
        old_global_threshold = system_config.queue_overflow_threshold
        old_submitter_threshold = system_config.per_submitter_queue_threshold
        try:
            system_config.queue_overflow_threshold = 10
            system_config.per_submitter_queue_threshold = 2
            queue = leanoj_module._LeanOJBrainstormSubmissionQueue(submitter_count=1)  # type: ignore[attr-defined]

            await queue.put((1, "submission one", {}))
            await queue.put((1, "submission two", {}))
            await queue.put((1, "submission three", {}))

            self.assertFalse(queue.should_pause_submitter(1))
            transitions = queue.refresh_pause_transitions()
            self.assertFalse(transitions["global_paused"])
            self.assertEqual(transitions["submitters_paused"], set())
        finally:
            system_config.queue_overflow_threshold = old_global_threshold
            system_config.per_submitter_queue_threshold = old_submitter_threshold

    async def test_brainstorm_dequeue_updates_submitter_pending_counts(self) -> None:
        old_global_threshold = system_config.queue_overflow_threshold
        old_submitter_threshold = system_config.per_submitter_queue_threshold
        try:
            system_config.queue_overflow_threshold = 10
            system_config.per_submitter_queue_threshold = 1
            coordinator = LeanOJCoordinator()
            queue = leanoj_module._LeanOJBrainstormSubmissionQueue(submitter_count=2)  # type: ignore[attr-defined]

            await queue.put((1, "submission one", {}))
            await queue.put((1, "submission two", {}))
            await queue.put((2, "submission three", {}))

            self.assertTrue(queue.should_pause_submitter(1))
            batch = await coordinator._dequeue_brainstorm_batch(queue, max_count=2)  # type: ignore[attr-defined]

            self.assertEqual(
                batch,
                [
                    (1, "submission one", {}),
                    (1, "submission two", {}),
                ],
            )
            self.assertEqual(queue.count_for_submitter(1), 0)
            self.assertEqual(queue.count_for_submitter(2), 1)
            self.assertFalse(queue.should_pause_submitter(1))
            self.assertEqual(queue.qsize(), 1)
        finally:
            system_config.queue_overflow_threshold = old_global_threshold
            system_config.per_submitter_queue_threshold = old_submitter_threshold

    async def test_accepted_brainstorm_proof_records_verified_subproof_context(self) -> None:
        request = _request()
        coordinator = await self._initialized_coordinator()

        await coordinator._record_accepted_brainstorm_proof(
            request,
            1,
            {
                "brainstorm_lean_proof": {
                    "theorem_statement": "True is true.",
                    "theorem_name": "brainstorm_true",
                    "formal_sketch": "Proof fragment from brainstorm.",
                    "lean_code": "import Mathlib\n\ntheorem brainstorm_true : True := by trivial",
                    "attempt_count": 2,
                }
            },
        )

        self.assertEqual(len(coordinator.get_state().verified_subproofs), 1)
        proof = coordinator.get_state().verified_subproofs[0]
        self.assertTrue(proof.verified)
        self.assertEqual(proof.attempts_used, 2)
        self.assertIn("brainstorm_true", proof.lean_code)

    async def test_final_loop_edits_master_proof_before_lean_verification(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir

        responses = [
            {
                "action": "edit_proof",
                "needs_more_time": True,
                "operation": "replace",
                "old_string": "sorry",
                "new_string": "simp",
                "reasoning": "First close the obvious placeholder, but keep editing time.",
            },
            {
                "action": "edit_proof",
                "needs_more_time": False,
                "operation": "replace",
                "old_string": "simp",
                "new_string": "rfl",
                "reasoning": "Use the final proof term and verify now.",
            },
        ]

        async def fake_call_json(_config, task_prefix, *_args, **_kwargs):
            if task_prefix == "leanoj_final_review":
                return {"solved": True, "reasoning": "final answer complete"}
            return responses.pop(0)

        class FakeLean:
            def __init__(self) -> None:
                self.calls = 0
                self.seen_code = ""

            async def check_proof(self, code: str, timeout: int = 120, **_kwargs):
                self.calls += 1
                self.seen_code = code
                return SimpleNamespace(success=True, error_output="")

        fake_lean = FakeLean()
        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: fake_lean  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(fake_lean.calls, 2)
                self.assertIn("rfl", fake_lean.seen_code)
                self.assertNotIn("simp", fake_lean.seen_code)
                self.assertEqual(coordinator.get_state().phase, "verified")
                self.assertEqual(coordinator.get_state().final_attempt_count, 1)
                self.assertEqual(coordinator.get_state().master_proof_version, 3)
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_loop_continues_when_final_review_rejects_lean_pass(self) -> None:
        request = _request()
        request.final_attempts_per_cycle = 1
        old_data_dir = system_config.data_dir

        async def fake_call_json(*_args, **_kwargs):
            return {
                "action": "edit_proof",
                "needs_more_time": False,
                "operation": "replace",
                "old_string": "sorry",
                "new_string": "rfl",
                "reasoning": "Lean should accept this equality proof.",
            }

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                return SimpleNamespace(
                    success=True,
                    error_output="",
                    diagnostic_output="Lean 4 accepted with an informational diagnostic.",
                    goal_states="",
                    raw_stderr="",
                )

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]

                async def reject_review(*_args, **_kwargs):
                    return (
                        False,
                        "The Lean proof checks, but it does not answer the actual problem prompt.",
                        "Lean 4 accepted with an informational diagnostic.",
                    )

                coordinator._review_final_solution_completion = reject_review  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "path_decision")
                self.assertFalse(coordinator.get_state().final_solution)
                self.assertIn("FINAL SOLUTION REVIEW REJECTED", coordinator._final_attempts[-1]["error_summary"])  # type: ignore[attr-defined]
                self.assertIn("informational diagnostic", coordinator._final_attempts[-1]["lean_feedback"])  # type: ignore[attr-defined]
                self.assertTrue(
                    any(
                        "actual problem prompt" in feedback.get("error_summary", "")
                        for feedback in coordinator._failed_feedback  # type: ignore[attr-defined]
                    )
                )
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_solution_review_prompt_includes_problem_template_code_and_lean_feedback(self) -> None:
        request = _request()
        coordinator = LeanOJCoordinator()
        captured_prompt = ""

        async def fake_call_json(_config, task_prefix, role_id, prompt, **_kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            self.assertEqual(task_prefix, "leanoj_final_review")
            self.assertEqual(role_id, "leanoj_final_solver")
            return {
                "solved": False,
                "continuation_feedback": "Give an explicit answer instead of a circular maximum.",
                "reasoning": "The formal code is evasive.",
            }

        coordinator._call_json = fake_call_json  # type: ignore[method-assign]
        solved, feedback, lean_feedback = await LeanOJCoordinator._review_final_solution_completion(
            coordinator,
            request,
            lean_code="import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
            final_solver_reasoning="This should close the template.",
            lean_result=SimpleNamespace(
                success=True,
                error_output="",
                diagnostic_output="Lean 4 accepted with a useful warning.",
                goal_states="",
                raw_stderr="",
            ),
        )

        self.assertFalse(solved)
        self.assertIn("explicit answer", feedback)
        self.assertIn("useful warning", lean_feedback)
        self.assertIn(request.user_prompt, captured_prompt)
        self.assertIn(request.lean_template, captured_prompt)
        self.assertIn("example : 1 = 1", captured_prompt)
        self.assertIn("Lean 4 accepted with a useful warning.", captured_prompt)
        self.assertIn("maximum/supremum over the same feasible set", captured_prompt)

    async def test_final_review_can_reject_evasive_lean_accepted_answer(self) -> None:
        request = _request()
        request.lean_template = (
            "import Mathlib.Data.Finset.Card\n"
            "import Mathlib.Order.Bounds.Defs\n\n"
            "def answer (n : ℕ) : ℕ := sorry\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | a = 0 }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := sorry"
        )
        request.final_attempts_per_cycle = 1
        old_data_dir = system_config.data_dir
        evasive_code = (
            "import Mathlib.Data.Finset.Card\n"
            "import Mathlib.Order.Bounds.Defs\n"
            "import Mathlib\n\n"
            "noncomputable def candidates (n : ℕ) : Finset ℕ := {0}\n"
            "noncomputable def answer (n : ℕ) : ℕ := (candidates n).max' (by simp [candidates])\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | a = 0 }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := by\n"
            "  exact ⟨by simp [S, answer, candidates], by intro y hy; simp [S, answer, candidates] at hy ⊢⟩"
        )

        async def fake_call_json(*_args, **_kwargs):
            return {"lean_code": evasive_code, "reasoning": "Lean accepts this maximum-based construction."}

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                return SimpleNamespace(success=True, error_output="", diagnostic_output="", goal_states="", raw_stderr="")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]

                async def reject_evasive_review(*_args, lean_code: str, **_kwargs):
                    self.assertIn("candidates", lean_code)
                    return False, "This defines `answer` by searching candidates instead of giving the requested formula.", ""

                coordinator._review_final_solution_completion = reject_evasive_review  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "path_decision")
                self.assertFalse(coordinator.get_state().final_solution)
                self.assertIn("searching candidates", coordinator._final_attempts[-1]["error_summary"])  # type: ignore[attr-defined]
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_loop_rejects_duplicate_master_proof_old_string(self) -> None:
        request = _request()
        request.lean_template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry\n\nexample : 2 = 2 := by\n  sorry"
        request.final_attempts_per_cycle = 1
        old_data_dir = system_config.data_dir

        async def fake_call_json(*_args, **_kwargs):
            return {
                "action": "edit_proof",
                "needs_more_time": False,
                "operation": "replace",
                "old_string": "sorry",
                "new_string": "rfl",
                "reasoning": "Ambiguous edit.",
            }

        class FakeLean:
            def __init__(self) -> None:
                self.calls = 0

            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                self.calls += 1
                return SimpleNamespace(success=True, error_output="")

        fake_lean = FakeLean()
        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: fake_lean  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(fake_lean.calls, 0)
                self.assertEqual(coordinator.get_state().final_attempt_count, 1)
                self.assertIn("appears 2 times", coordinator._final_attempts[-1]["error_summary"])  # type: ignore[attr-defined]
                self.assertEqual(coordinator.get_state().phase, "path_decision")
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_loop_rejects_nonprogressive_shortening_before_write(self) -> None:
        request = _request()
        request.lean_template = (
            "import Mathlib\n\n"
            "theorem helper : True := by\n"
            "  trivial\n\n"
            "example : True := by\n"
            "  exact helper"
        )
        request.final_attempts_per_cycle = 1
        old_data_dir = system_config.data_dir
        events: list[tuple[str, dict]] = []

        async def fake_call_json(_config, task_prefix, role_id, _prompt, **_kwargs):
            if task_prefix == "leanoj_master_proof_edit_val":
                self.assertEqual(role_id, "leanoj_master_proof_edit_validator")
                return {
                    "decision": "reject",
                    "reasoning": "The edit deletes the proved helper and returns to a placeholder.",
                    "feedback_to_submitter": "Restore theorem helper or replace it with an equivalent proof before shortening.",
                }
            return {
                "action": "edit_proof",
                "needs_more_time": False,
                "operation": "full_content",
                "new_string": "import Mathlib\n\nexample : True := by\n  sorry",
                "reasoning": "Shorten the file by restarting from the goal.",
            }

        class FakeLean:
            def __init__(self) -> None:
                self.calls = 0

            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                self.calls += 1
                raise AssertionError("Lean should not run after validator rejects the shortening edit")

        fake_lean = FakeLean()
        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]

                async def capture_broadcast(event: str, data: dict | None = None):
                    events.append((event, data or {}))

                coordinator._broadcast = capture_broadcast  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: fake_lean  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(fake_lean.calls, 0)
                self.assertEqual(await coordinator._read_master_proof(), request.lean_template.strip())  # type: ignore[attr-defined]
                self.assertEqual(coordinator.get_state().final_attempt_count, 1)
                self.assertEqual(coordinator.get_state().phase, "path_decision")
                self.assertIn("Restore theorem helper", coordinator._final_attempts[-1]["error_summary"])  # type: ignore[attr-defined]
                self.assertTrue(
                    any(
                        "Restore theorem helper" in feedback.get("error_summary", "")
                        for feedback in coordinator._failed_feedback  # type: ignore[attr-defined]
                    )
                )
                self.assertIn("leanoj_master_proof_edit_rejected", [event for event, _data in events])
                edits = await coordinator.get_master_proof_edit_summaries(limit=1)
                self.assertFalse(edits["edits"][0]["accepted"])
                self.assertIn("Restore theorem helper", edits["edits"][0]["validator_feedback"])
                self.assertGreater(edits["edits"][0]["shortening_metrics"]["line_delta_removed"], 0)
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_loop_allows_validator_accepted_shortening(self) -> None:
        request = _request()
        request.lean_template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        old_data_dir = system_config.data_dir

        shortened_code = "import Mathlib\n\nexample : 1 = 1 := by\n  rfl"

        async def fake_call_json(_config, task_prefix, role_id, _prompt, **_kwargs):
            if task_prefix == "leanoj_master_proof_edit_val":
                self.assertEqual(role_id, "leanoj_master_proof_edit_validator")
                return {
                    "decision": "accept",
                    "reasoning": "The shorter proof removes only redundant helper scaffolding and keeps the solved template.",
                    "feedback_to_submitter": "",
                }
            return {
                "action": "edit_proof",
                "needs_more_time": False,
                "operation": "full_content",
                "new_string": shortened_code,
                "reasoning": "Replace the verbose draft with the direct final proof.",
            }

        class FakeLean:
            def __init__(self) -> None:
                self.calls = 0
                self.seen_code = ""

            async def check_proof(self, code: str, timeout: int = 120, **_kwargs):
                self.calls += 1
                self.seen_code = code
                return SimpleNamespace(success=True, error_output="", diagnostic_output="", goal_states="", raw_stderr="")

        fake_lean = FakeLean()
        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                await coordinator._write_master_proof(  # type: ignore[attr-defined]
                    "import Mathlib\n\n"
                    "theorem helper : 1 = 1 := by\n"
                    "  rfl\n\n"
                    "example : 1 = 1 := by\n"
                    "  exact helper",
                    summary="verbose seed",
                )
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: fake_lean  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(fake_lean.calls, 1)
                self.assertEqual(fake_lean.seen_code, shortened_code)
                self.assertEqual(coordinator.get_state().phase, "verified")
                self.assertEqual(await coordinator._read_master_proof(), shortened_code)  # type: ignore[attr-defined]
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_final_loop_rejects_phase_transition_action_as_invalid_edit(self) -> None:
        request = _request()
        request.final_attempts_per_cycle = 3
        old_data_dir = system_config.data_dir
        call_count = 0

        async def fake_call_json(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "action": "phase_transition",
                "reasoning": "Trying to leave final mode should be rejected.",
            }

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                raise AssertionError("Lean should not be called for an invalid final solver action")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "path_decision")
                self.assertEqual(coordinator.get_state().current_path_decision, "need_more_brainstorming")
                self.assertEqual(call_count, 3)
                self.assertEqual(coordinator.get_state().final_attempt_count, 3)
                self.assertIn("Invalid final solver action", coordinator._final_attempts[-1]["error_summary"])  # type: ignore[attr-defined]
                self.assertNotIn("brainstorm", coordinator._final_attempts[-1]["error_summary"].lower())  # type: ignore[attr-defined]
                self.assertFalse(coordinator.get_state().master_proof_last_stuck_reason)
                self.assertEqual(await coordinator._read_master_proof(), request.lean_template.strip())  # type: ignore[attr-defined]
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_skip_brainstorm_enters_final_loop_without_path_decision(self) -> None:
        request = _request()
        coordinator = await self._initialized_coordinator()
        coordinator._state.phase = "recursive_brainstorm"  # type: ignore[attr-defined]
        coordinator._state.skip_brainstorm_requested = True  # type: ignore[attr-defined]
        final_loop_calls = 0

        async def fake_final_proof_loop(_request):
            nonlocal final_loop_calls
            final_loop_calls += 1
            coordinator._state.phase = "verified"  # type: ignore[attr-defined]

        async def fail_path_decision(_request):
            raise AssertionError("Path decision should not run after brainstorm skip")

        coordinator._final_proof_loop = fake_final_proof_loop  # type: ignore[method-assign]
        coordinator._path_decision_phase = fail_path_decision  # type: ignore[method-assign]

        await coordinator._run_workflow(request)  # type: ignore[attr-defined]

        self.assertEqual(final_loop_calls, 1)
        self.assertFalse(coordinator.get_state().skip_brainstorm_requested)
        self.assertEqual(coordinator.get_state().current_path_decision, "solve_final_now")
        self.assertEqual(coordinator.get_state().phase, "verified")

    async def test_forced_final_phase_runs_before_next_path_decision(self) -> None:
        request = _request()
        coordinator = await self._initialized_coordinator()
        coordinator._state.phase = "recursive_brainstorm"  # type: ignore[attr-defined]
        final_loop_calls = 0

        async def fake_recursive_brainstorm(_request):
            coordinator._state.phase = "final_proof_loop"  # type: ignore[attr-defined]
            coordinator._state.user_forced_final_cycle = True  # type: ignore[attr-defined]

        async def fake_final_proof_loop(_request):
            nonlocal final_loop_calls
            final_loop_calls += 1
            coordinator._state.user_forced_final_cycle = False  # type: ignore[attr-defined]
            coordinator._state.phase = "verified"  # type: ignore[attr-defined]

        async def fail_path_decision(_request):
            raise AssertionError("Path decision should not run while forced final cycle is active")

        coordinator._recursive_brainstorm_phase = fake_recursive_brainstorm  # type: ignore[method-assign]
        coordinator._final_proof_loop = fake_final_proof_loop  # type: ignore[method-assign]
        coordinator._path_decision_phase = fail_path_decision  # type: ignore[method-assign]

        await coordinator._run_workflow(request)  # type: ignore[attr-defined]

        self.assertEqual(final_loop_calls, 1)
        self.assertEqual(coordinator.get_state().phase, "verified")

    async def test_forced_final_cycle_uses_all_attempts_before_path(self) -> None:
        request = _request().model_copy(update={"final_attempts_per_cycle": 3})
        coordinator = await self._initialized_coordinator()
        coordinator._state.user_forced_final_cycle = True  # type: ignore[attr-defined]
        call_count = 0

        async def fake_call_json(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "action": "phase_transition",
                "reasoning": f"invalid action {call_count}",
            }

        coordinator._call_json = fake_call_json  # type: ignore[method-assign]

        await coordinator._final_proof_loop(request)

        self.assertEqual(call_count, 3)
        self.assertEqual(coordinator.get_state().final_attempt_count, 3)
        self.assertFalse(coordinator.get_state().user_forced_final_cycle)
        self.assertEqual(coordinator.get_state().phase, "path_decision")
        self.assertEqual(coordinator.get_state().current_path_decision, "need_more_brainstorming")

    async def test_recursive_brainstorm_targets_current_working_proof_attempt(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir
        old_allocate = leanoj_module.leanoj_context_manager.allocate_context
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._state.phase = "recursive_brainstorm"  # type: ignore[attr-defined]
                coordinator._state.selected_topic = "Original broad equality topic"
                await coordinator._write_master_proof(  # type: ignore[attr-defined]
                    "import Mathlib\n\nexample : 1 = 1 := by\n  have h : 1 = 1 := by\n    sorry",
                    summary="latest draft",
                )
                coordinator._final_attempts.append(  # type: ignore[attr-defined]
                    {
                        "request": "final Proof Solver solution",
                        "error_summary": "unsolved goals at h",
                        "lean_code": "bad final",
                    }
                )
                await coordinator._set_current_working_proof_attempt(  # type: ignore[attr-defined]
                    trigger="final_solver_stuck",
                    requested_path="need_more_brainstorming",
                    stuck_reason="Need a way to close h.",
                )

                async def fake_allocate_context(**kwargs):
                    packet = kwargs.get("current_working_proof_attempt") or {}
                    return LeanOJContextAllocation(
                        current_working_proof_attempt=(
                            "CURRENT WORKING PROOF ATTEMPT\n"
                            f"{packet.get('master_proof', '')}\n"
                            f"{packet.get('recent_final_attempts', '')}"
                        )
                    )

                prompts = []

                async def fake_call_json(_config, _task_prefix, _role_id, prompt, **_kwargs):
                    prompts.append(prompt)
                    coordinator._stop_event.set()  # type: ignore[attr-defined]
                    return {"submission": "Use rfl to close the local equality.", "reasoning": "direct repair"}

                leanoj_module.leanoj_context_manager.allocate_context = fake_allocate_context  # type: ignore[assignment]
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                queue = leanoj_module._LeanOJBrainstormSubmissionQueue(submitter_count=1)  # type: ignore[attr-defined]

                await coordinator._brainstorm_submitter_loop(request, 1, request.brainstorm_submitters[0], queue)  # type: ignore[attr-defined]

                self.assertTrue(prompts)
                prompt = prompts[0]
                self.assertIn("ACTIVE TOPIC:\nRepair and complete the current Proof Solver master proof attempt", prompt)
                self.assertNotIn("ACTIVE TOPIC:\nOriginal broad equality topic", prompt)
                self.assertIn("CURRENT WORKING PROOF ATTEMPT", prompt)
                self.assertIn("have h : 1 = 1", prompt)
                self.assertIn("unsolved goals at h", prompt)
            finally:
                leanoj_module.leanoj_context_manager.allocate_context = old_allocate  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_master_proof_direct_context_overflow_raises(self) -> None:
        request = _request()
        request.final_solver.context_window = 28000
        request.final_solver.max_output_tokens = 25000
        coordinator = await self._initialized_coordinator()
        large_proof = "\n".join(
            [
                "import Mathlib",
                "",
                *[f"def filler_{index} : Nat := {index}" for index in range(1, 900)],
                "example : 1 = 1 := by",
                "  sorry",
                *[f"def tail_filler_{index} : Nat := {index}" for index in range(900, 1800)],
            ]
        )

        with self.assertRaisesRegex(LeanOJConfigurationError, "MANDATORY DIRECT CONTEXT OVERFLOW"):
            coordinator._build_master_proof_direct_context(  # type: ignore[attr-defined]
                large_proof,
                request,
                context_blocks={},
            )

    async def test_master_proof_draft_and_edit_summaries_are_read_on_demand(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                await coordinator._write_master_proof(  # type: ignore[attr-defined]
                    "import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
                    summary="draft ready",
                )
                await coordinator._append_master_proof_edit(  # type: ignore[attr-defined]
                    {
                        "action": "edit_proof",
                        "operation": "replace",
                        "old_string": "sorry",
                        "new_string": "rfl",
                        "reasoning": "Closed the proof.",
                        "accepted": True,
                    }
                )

                draft = await coordinator.get_master_proof_draft()
                edits = await coordinator.get_master_proof_edit_summaries(limit=1)

                self.assertTrue(draft["exists"])
                self.assertIn("rfl", draft["content"])
                self.assertEqual(draft["metadata"]["version"], 1)
                self.assertEqual(edits["total_edits"], 1)
                self.assertEqual(len(edits["edits"]), 1)
                self.assertEqual(edits["edits"][0]["new_string_preview"], "rfl")
            finally:
                system_config.data_dir = old_data_dir

    async def test_master_proof_edit_log_compacts_to_snapshot(self) -> None:
        old_data_dir = system_config.data_dir
        old_limit = leanoj_module._MASTER_PROOF_EDIT_LOG_COMPACT_RECORD_LIMIT
        old_keep = leanoj_module._MASTER_PROOF_EDIT_LOG_RECENT_RECORDS_TO_KEEP
        leanoj_module._MASTER_PROOF_EDIT_LOG_COMPACT_RECORD_LIMIT = 3
        leanoj_module._MASTER_PROOF_EDIT_LOG_RECENT_RECORDS_TO_KEEP = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                await coordinator._write_master_proof("import Mathlib", summary="seed")  # type: ignore[attr-defined]
                for index in range(5):
                    await coordinator._append_master_proof_edit(  # type: ignore[attr-defined]
                        {
                            "action": "edit_proof",
                            "operation": "insert_after",
                            "old_string": "import Mathlib",
                            "new_string": f"def helper_{index} : Nat := {index}",
                            "reasoning": f"edit {index}",
                            "accepted": True,
                        }
                    )

                records = coordinator._read_master_proof_edit_records()  # type: ignore[attr-defined]
                snapshot_path = coordinator._master_proof_snapshot_log_path()  # type: ignore[attr-defined]

                self.assertLessEqual(len(records), 3)
                self.assertTrue(snapshot_path.exists())
                self.assertIn("master_proof_edit_log_compaction", snapshot_path.read_text(encoding="utf-8"))
            finally:
                leanoj_module._MASTER_PROOF_EDIT_LOG_COMPACT_RECORD_LIMIT = old_limit
                leanoj_module._MASTER_PROOF_EDIT_LOG_RECENT_RECORDS_TO_KEEP = old_keep
                system_config.data_dir = old_data_dir

    async def test_master_proof_progress_watchdog_returns_to_brainstorming(self) -> None:
        request = _request()
        request.final_attempts_per_cycle = 3
        old_data_dir = system_config.data_dir
        old_limit = leanoj_module._MASTER_PROOF_NO_PROGRESS_LIMIT
        leanoj_module._MASTER_PROOF_NO_PROGRESS_LIMIT = 2
        call_count = 0

        async def fake_call_json(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "action": "edit_proof",
                "needs_more_time": True,
                "operation": "insert_after",
                "old_string": "import Mathlib",
                "new_string": f"def repeated_region_{call_count} : Nat := {call_count}",
                "reasoning": "Keep expanding the same import anchor.",
            }

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                return SimpleNamespace(success=True, error_output="")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "recursive_brainstorm")
                self.assertEqual(coordinator.get_state().current_path_decision, "need_more_brainstorming")
                self.assertEqual(coordinator.get_state().final_attempt_count, 1)
                self.assertIn("same proof region", coordinator.get_state().master_proof_last_stuck_reason)
                self.assertGreaterEqual(call_count, 1)
            finally:
                leanoj_module._MASTER_PROOF_NO_PROGRESS_LIMIT = old_limit
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_leanoj_master_proof_routes_return_draft_and_edits(self) -> None:
        from backend.api.routes import leanoj as leanoj_route_module

        old_data_dir = system_config.data_dir
        old_route_coordinator = leanoj_route_module.leanoj_coordinator
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                await coordinator._write_master_proof("import Mathlib\n\nexample : True := by\n  trivial", summary="api")  # type: ignore[attr-defined]
                await coordinator._append_master_proof_edit(  # type: ignore[attr-defined]
                    {
                        "action": "edit_proof",
                        "operation": "full_content",
                        "new_string": "import Mathlib\n\nexample : True := by\n  trivial",
                        "reasoning": "API route test.",
                        "accepted": True,
                    }
                )
                leanoj_route_module.leanoj_coordinator = coordinator

                draft = await leanoj_route_module.get_leanoj_master_proof()
                edits = await leanoj_route_module.get_leanoj_master_proof_edits(limit=5)

                self.assertTrue(draft["exists"])
                self.assertIn("trivial", draft["content"])
                self.assertEqual(edits["total_edits"], 1)
            finally:
                leanoj_route_module.leanoj_coordinator = old_route_coordinator
                system_config.data_dir = old_data_dir

    async def test_leanoj_library_exposes_shared_proof_tiers(self) -> None:
        from backend.api.routes import leanoj as leanoj_route_module

        payload = {
            "session_id": "leanoj_session",
            "user_prompt": "Prove one equals one.",
            "selected_topic": "Equality",
            "phase": "verified",
            "final_solution": "import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
            "final_proof_id": "proof_final",
            "final_novel": True,
            "final_novelty_tier": "mathematical_discovery",
            "final_novelty_reasoning": "Final proof is novel in context.",
            "verified_subproofs": [
                {
                    "subproof_id": "subproof_1",
                    "request": "Show reflexivity.",
                    "verified": True,
                    "lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
                    "proof_id": "proof_sub",
                    "novel": True,
                    "novelty_tier": "novel_formulation",
                    "novelty_reasoning": "Subproof formalization is useful.",
                }
            ],
        }

        proofs = leanoj_route_module._extract_leanoj_proofs(payload)

        final = next(proof for proof in proofs if proof["proof_kind"] == "final")
        subproof = next(proof for proof in proofs if proof["proof_kind"] == "subproof")
        self.assertEqual(final["shared_proof_id"], "proof_final")
        self.assertEqual(final["novelty_tier"], "mathematical_discovery")
        self.assertEqual(subproof["shared_proof_id"], "proof_sub")
        self.assertEqual(subproof["novelty_tier"], "novel_formulation")

    async def test_skip_brainstorm_sets_state_flag(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()

                await coordinator.skip_brainstorm()

                self.assertTrue(coordinator.get_state().skip_brainstorm_requested)
            finally:
                system_config.data_dir = old_data_dir

    async def test_skip_brainstorm_is_consumed_once(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                await coordinator.skip_brainstorm()

                await coordinator._brainstorm_until_path_check(  # type: ignore[attr-defined]
                    _request(),
                    max_accepts=1,
                    sufficiency_interval=1,
                    force_after_max=True,
                )

                self.assertFalse(coordinator.get_state().skip_brainstorm_requested)
            finally:
                system_config.data_dir = old_data_dir

    async def test_force_brainstorm_resets_recursive_acceptance_window(self) -> None:
        coordinator = await self._initialized_coordinator()
        state = coordinator.get_state()
        state.phase = "final_proof_loop"
        state.active_brainstorm_phase = "recursive_brainstorm"
        state.active_brainstorm_start_count = 10
        state.brainstorm_acceptance_events = 14

        await coordinator.force_brainstorm()
        consumed = await coordinator._consume_force_brainstorm()  # type: ignore[attr-defined]

        self.assertTrue(consumed)
        self.assertEqual(state.phase, "recursive_brainstorm")
        self.assertEqual(state.active_brainstorm_phase, "")
        self.assertEqual(state.active_brainstorm_start_count, 14)
        self.assertEqual(
            coordinator._get_brainstorm_acceptance_start("recursive_brainstorm"),  # type: ignore[attr-defined]
            14,
        )

    async def test_skip_brainstorm_prevents_recursive_brainstorm_start(self) -> None:
        coordinator = await self._initialized_coordinator()
        coordinator._state.phase = "path_decision"  # type: ignore[attr-defined]
        coordinator._state.skip_brainstorm_requested = True  # type: ignore[attr-defined]

        async def fail_brainstorm_until_path_check(*_args, **_kwargs):
            raise AssertionError("Recursive brainstorm should not start after skip brainstorm")

        coordinator._brainstorm_until_path_check = fail_brainstorm_until_path_check  # type: ignore[method-assign]

        await coordinator._recursive_brainstorm_phase(_request())  # type: ignore[attr-defined]

        self.assertFalse(coordinator.get_state().skip_brainstorm_requested)
        self.assertTrue(coordinator.get_state().user_forced_final_cycle)
        self.assertEqual(coordinator.get_state().phase, "final_proof_loop")
        self.assertEqual(coordinator.get_state().current_path_decision, "solve_final_now")

    async def test_path_decision_uses_final_solver_actor_when_final_path_available(self) -> None:
        coordinator = await self._initialized_coordinator()
        request = _request().model_copy(
            update={
                "path_decider": _role().model_copy(update={"model_id": "legacy-path-model"}),
                "final_solver": _role().model_copy(update={"model_id": "final-model"}),
            }
        )
        calls = []
        context_models = []

        async def fake_build_context_blocks(_request, config, **_kwargs):
            context_models.append(config.model_id)
            return {}

        async def fake_call_json(config, task_prefix, role_id, _prompt, **_kwargs):
            calls.append({"model_id": config.model_id, "task_prefix": task_prefix, "role_id": role_id})
            if task_prefix == "leanoj_path":
                return {"path": "solve_final_now", "reasoning": "ready for final proof"}
            if role_id == "leanoj_path_validator":
                return {"decision": "accept", "reasoning": "valid", "summary": ""}
            return {}

        coordinator._build_context_blocks = fake_build_context_blocks  # type: ignore[method-assign]
        coordinator._call_json = fake_call_json  # type: ignore[method-assign]

        decision = await coordinator._path_decision_phase(request)  # type: ignore[attr-defined]

        self.assertEqual(decision, "solve_final_now")
        self.assertEqual(calls[0]["model_id"], "final-model")
        self.assertEqual(calls[0]["role_id"], "leanoj_final_solver")
        self.assertEqual(context_models[0], "final-model")

    def test_path_decision_actor_falls_back_to_topic_generator_without_final_option(self) -> None:
        request = _request().model_copy(
            update={
                "topic_generator": _role().model_copy(update={"model_id": "topic-model"}),
                "final_solver": _role().model_copy(update={"model_id": "final-model"}),
            }
        )

        actor, role_id = LeanOJCoordinator._path_decision_actor(  # type: ignore[attr-defined]
            request,
            valid_paths=("need_more_brainstorming",),
        )

        self.assertEqual(actor.model_id, "topic-model")
        self.assertEqual(role_id, "leanoj_topic_generator")

    def test_leanoj_path_boost_category_is_absorbed_into_final_solver(self) -> None:
        manager = BoostManager()

        self.assertEqual(manager._extract_role_prefix("leanoj_path_003"), "leanoj_final")
        self.assertEqual(manager._canonical_category("leanoj_path"), "leanoj_final")

    async def test_final_loop_retries_after_malformed_model_output(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir

        calls = 0

        async def fake_call_json(_config, task_prefix, *_args, **_kwargs):
            nonlocal calls
            if task_prefix == "leanoj_final_review":
                return {"solved": True, "reasoning": "final answer complete"}
            calls += 1
            if calls == 1:
                raise ValueError("No JSON found in response")
            return {"lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  rfl", "reasoning": "retry"}

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                return SimpleNamespace(success=True, error_output="")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                self.assertEqual(coordinator.get_state().phase, "verified")
                self.assertEqual(coordinator.get_state().final_attempt_count, 2)
                self.assertEqual(calls, 2)
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_placeholder_scaffold_is_saved_as_partial_not_verified(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, *, allow_placeholders: bool = False):
                if not allow_placeholders:
                    return SimpleNamespace(success=False, error_output="placeholders were not allowed")
                return SimpleNamespace(success=True, error_output="", goal_states="", raw_stderr="")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                result = await coordinator._check_proof_and_capture_partial(  # type: ignore[attr-defined]
                    request,
                    "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
                    target="final",
                    attempt_number=1,
                    proof_request="final Proof Solver solution",
                    reasoning="scaffold",
                )

                self.assertFalse(result.success)
                self.assertIn("PARTIAL PROOF SAVED", result.error_output)
                partials = coordinator.get_status()["partial_proofs"]
                self.assertEqual(len(partials), 1)
                self.assertEqual(partials[0]["placeholder_tokens"], ["sorry"])
                self.assertFalse(coordinator.get_state().final_solution)
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_unrelated_final_placeholder_scaffold_is_not_saved(self) -> None:
        request = _request()
        old_data_dir = system_config.data_dir

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, *, allow_placeholders: bool = False):
                return SimpleNamespace(success=True, error_output="", goal_states="", raw_stderr="")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                result = await coordinator._check_proof_and_capture_partial(  # type: ignore[attr-defined]
                    request,
                    "import Mathlib\n\ntheorem unrelated : True := by\n  sorry",
                    target="final",
                    attempt_number=1,
                    proof_request="final Proof Solver solution",
                    reasoning="bad scaffold",
                )

                self.assertFalse(result.success)
                self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", result.error_output)
                self.assertEqual(coordinator.get_status()["partial_proofs"], [])
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_restore_loads_partial_proof_database_records(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                record = {
                    "session_id": coordinator.get_state().session_id,
                    "attempt": 7,
                    "target": "final",
                    "request": "final Proof Solver solution",
                    "placeholder_tokens": ["sorry"],
                    "lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
                }
                await coordinator._append_partial_proof_database(record)  # type: ignore[attr-defined]

                payload = coordinator.get_status()
                payload["request"] = _request().model_dump(mode="json")
                payload["partial_proofs"] = []

                restored = LeanOJCoordinator()
                restored._restore_from_payload(payload)  # type: ignore[attr-defined]

                partials = restored.get_status()["partial_proofs"]
                self.assertEqual(len(partials), 1)
                self.assertEqual(partials[0]["attempt"], 7)
            finally:
                system_config.data_dir = old_data_dir

    async def test_brainstorm_submitters_run_in_parallel_and_batch_validate(self) -> None:
        request = _request()
        request.brainstorm_submitters = [_role(), _role(), _role()]
        coordinator = await self._initialized_coordinator()

        never_finish = leanoj_module.asyncio.Event()
        submitter_calls: list[str] = []
        validator_prompts: list[str] = []
        per_submitter_counts: dict[str, int] = {}

        async def fake_call_json(_config, _task_prefix, role_id, prompt, **_kwargs):
            if role_id.startswith("leanoj_brainstorm_submitter_"):
                submitter_calls.append(role_id)
                per_submitter_counts[role_id] = per_submitter_counts.get(role_id, 0) + 1
                if role_id == "leanoj_brainstorm_submitter_2" or per_submitter_counts[role_id] > 1:
                    await never_finish.wait()
                return {"submission": f"{role_id} useful idea"}

            if role_id == "leanoj_brainstorm_validator":
                validator_prompts.append(prompt)
                batch_size = prompt.count("SUBMISSION ")
                if batch_size:
                    return {
                        "decisions": [
                            {
                                "submission_number": index,
                                "decision": "accept",
                                "reasoning": "useful",
                                "summary": "accepted",
                            }
                            for index in range(1, batch_size + 1)
                        ]
                    }
                return {"decision": "accept", "reasoning": "useful", "summary": "accepted"}

            return {"enough": False}

        coordinator._call_json = fake_call_json  # type: ignore[method-assign]

        await leanoj_module.asyncio.wait_for(
            coordinator._brainstorm_until_path_check(  # type: ignore[attr-defined]
                request,
                max_accepts=2,
                sufficiency_interval=10,
                force_after_max=True,
            ),
            timeout=2,
        )

        self.assertIn("leanoj_brainstorm_submitter_2", submitter_calls)
        self.assertEqual(coordinator.get_state().accepted_brainstorm_count, 2)
        self.assertTrue(any("SUBMISSION 1:" in prompt and "SUBMISSION 2:" in prompt for prompt in validator_prompts))

    def test_template_mismatch_rejects_unrelated_compiling_code(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        unrelated = "import Mathlib\n\ntheorem unrelated : True := by\n  trivial"
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, unrelated)

        self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", error)

    def test_template_match_accepts_filled_hole(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        solved = "import Mathlib\n\nexample : 1 = 1 := by\n  rfl"
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, solved)

        self.assertEqual(error, "")

    def test_template_match_allows_answer_and_theorem_hole_replacement(self) -> None:
        template = (
            "import Mathlib.Data.Finset.Card\n"
            "import Mathlib.Order.Bounds.Defs\n\n"
            "def answer (n : ℕ) : ℕ := sorry\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | a = 0 }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := sorry"
        )
        solved = (
            "import Mathlib.Data.Finset.Card\n"
            "import Mathlib.Order.Bounds.Defs\n"
            "import Mathlib\n\n"
            "open Classical\n\n"
            "noncomputable def answer (n : ℕ) : ℕ := sSup (S n)\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | a = 0 }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := by\n"
            "  sorry"
        )
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, solved)

        self.assertEqual(error, "")

    def test_template_match_allows_open_classical_in_noncomputable_hole(self) -> None:
        template = "import Mathlib\n\ndef answer (n : ℕ) : ℕ := sorry"
        solved = "import Mathlib\n\nopen Classical in\nnoncomputable def answer (n : ℕ) : ℕ := sSup ({0} : Set ℕ)"
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, solved)

        self.assertEqual(error, "")

    def test_template_mismatch_rejects_changed_fixed_definition(self) -> None:
        template = (
            "import Mathlib\n\n"
            "def answer (n : ℕ) : ℕ := sorry\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | a = 0 }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := sorry"
        )
        changed = (
            "import Mathlib\n\n"
            "def answer (n : ℕ) : ℕ := 0\n\n"
            "def S (n : ℕ) : Set ℕ := { a : ℕ | True }\n\n"
            "theorem solution (n : ℕ) (hn : n > 0) : IsGreatest (S n) (answer n) := by\n"
            "  sorry"
        )
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, changed)

        self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", error)

    def test_template_mismatch_rejects_changed_fixed_instance(self) -> None:
        template = (
            "import Mathlib\n\n"
            "instance : Inhabited ℕ := ⟨0⟩\n\n"
            "example : 1 = 1 := by\n  sorry"
        )
        changed = (
            "import Mathlib\n\n"
            "instance : Inhabited ℕ := ⟨1⟩\n\n"
            "example : 1 = 1 := by\n  rfl"
        )
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, changed)

        self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", error)

    def test_template_mismatch_rejects_changed_theorem_target(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        changed = "import Mathlib\n\nexample : True := by\n  trivial"
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, changed)

        self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", error)

    def test_template_mismatch_rejects_import_only_in_comment(self) -> None:
        template = "import Mathlib.Data.Finset.Card\n\nexample : 1 = 1 := by\n  sorry"
        changed = "-- import Mathlib.Data.Finset.Card\nimport Mathlib\n\nexample : 1 = 1 := by\n  rfl"
        error = LeanOJCoordinator._validate_final_solution_matches_template(template, changed)

        self.assertIn("PROOF SOLVER TEMPLATE MISMATCH", error)

    def test_integrity_rejects_new_axiom_device(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        fake_solution = (
            "import Mathlib\n\n"
            "axiom fakeGoal : 1 = 1\n\n"
            "example : 1 = 1 := by\n"
            "  exact fakeGoal"
        )
        error = LeanOJCoordinator._validate_final_solution_integrity(template, fake_solution)

        self.assertIn("PROOF SOLVER FORBIDDEN PROOF DEVICE", error)

    def test_integrity_rejects_new_attributed_axiom_device(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        fake_solution = (
            "import Mathlib\n\n"
            "@[simp] axiom fakeGoal : 1 = 1\n\n"
            "example : 1 = 1 := by\n"
            "  exact fakeGoal"
        )
        error = LeanOJCoordinator._validate_final_solution_integrity(template, fake_solution)

        self.assertIn("PROOF SOLVER FORBIDDEN PROOF DEVICE", error)

    def test_integrity_rejects_new_escaped_axiom_device(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        fake_solution = (
            "import Mathlib\n\n"
            "axiom «fake goal» : 1 = 1\n\n"
            "example : 1 = 1 := by\n"
            "  exact «fake goal»"
        )
        error = LeanOJCoordinator._validate_final_solution_integrity(template, fake_solution)

        self.assertIn("PROOF SOLVER FORBIDDEN PROOF DEVICE", error)

    def test_integrity_rejects_parenthesized_constant_device(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        fake_solution = (
            "import Mathlib\n\n"
            "constant (fakeGoal : 1 = 1)\n\n"
            "example : 1 = 1 := by\n"
            "  exact fakeGoal"
        )
        error = LeanOJCoordinator._validate_final_solution_integrity(template, fake_solution)

        self.assertIn("PROOF SOLVER FORBIDDEN PROOF DEVICE", error)

    def test_integrity_allows_template_existing_constant(self) -> None:
        template = "import Mathlib\n\nconstant h : 1 = 1\n\nexample : 1 = 1 := by\n  sorry"
        solved = "import Mathlib\n\nconstant h : 1 = 1\n\nexample : 1 = 1 := by\n  exact h"
        error = LeanOJCoordinator._validate_final_solution_integrity(template, solved)

        self.assertEqual(error, "")

    def test_subproof_integrity_rejects_new_axiom_device(self) -> None:
        template = "import Mathlib\n\nexample : 1 = 1 := by\n  sorry"
        fake_subproof = "import Mathlib\n\naxiom fakeLemma : 1 = 1\n\ntheorem helper : 1 = 1 := fakeLemma"
        error = LeanOJCoordinator._validate_no_new_declaration_devices(
            template,
            fake_subproof,
            target="subproof",
        )

        self.assertIn("PROOF SOLVER FORBIDDEN PROOF DEVICE", error)

    async def test_initialize_rejects_missing_role_model(self) -> None:
        request = _request()
        request.final_solver.model_id = ""
        coordinator = LeanOJCoordinator()

        with self.assertRaisesRegex(ValueError, "final_solver"):
            await coordinator.initialize(request)

    async def test_call_json_missing_model_is_non_retryable_configuration_error(self) -> None:
        coordinator = await self._initialized_coordinator()

        with self.assertRaises(LeanOJConfigurationError):
            await coordinator._call_json(  # type: ignore[attr-defined]
                LeanOJRoleConfig(model_id=""),
                "leanoj_final",
                "leanoj_final_solver",
                "{}",
            )

    async def test_call_json_keeps_retrying_malformed_json_until_success(self) -> None:
        coordinator = await self._initialized_coordinator()
        old_generate_completion = leanoj_module.api_client_manager.generate_completion
        old_sleep = leanoj_module.asyncio.sleep
        calls = 0
        prompts: list[str] = []

        async def fake_generate_completion(**kwargs):
            nonlocal calls
            calls += 1
            prompts.append(kwargs["messages"][0]["content"])
            if calls < 5:
                return {"choices": [{"message": {"content": '{"decisions": ['}}]}
            return {"choices": [{"message": {"content": '{"decision": "accept", "reasoning": "ok"}'}}]}

        async def noop_sleep(*_args, **_kwargs):
            return None

        leanoj_module.api_client_manager.generate_completion = fake_generate_completion  # type: ignore[assignment]
        leanoj_module.asyncio.sleep = noop_sleep  # type: ignore[assignment]
        try:
            result = await coordinator._call_json(  # type: ignore[attr-defined]
                _role(),
                "leanoj_brainstorm_val",
                "leanoj_brainstorm_validator",
                "Return the requested JSON.",
            )
        finally:
            leanoj_module.api_client_manager.generate_completion = old_generate_completion  # type: ignore[assignment]
            leanoj_module.asyncio.sleep = old_sleep  # type: ignore[assignment]

        self.assertEqual(result["decision"], "accept")
        self.assertEqual(calls, 5)
        self.assertTrue(any("INVALID_OR_TRUNCATED_JSON" in prompt for prompt in prompts[1:]))
        self.assertTrue(
            any(
                feedback.get("role_id") == "leanoj_brainstorm_validator"
                for feedback in coordinator.get_status()["failed_feedback"]
            )
        )

    async def test_error_resume_prefers_existing_master_proof_loop(self) -> None:
        coordinator = await self._initialized_coordinator()
        state = coordinator.get_state()
        state.phase = "error"
        state.last_active_phase = "recursive_brainstorm"
        state.master_proof_initialized = True
        state.master_proof_version = 2

        self.assertEqual(coordinator._infer_resume_phase(), "final_proof_loop")  # type: ignore[attr-defined]

    async def test_stop_handles_main_task_cleared_during_timeout(self) -> None:
        coordinator = LeanOJCoordinator()

        async def noop(*_args, **_kwargs):
            return None

        coordinator._persist_state = noop  # type: ignore[method-assign]
        coordinator._broadcast = noop  # type: ignore[method-assign]
        task = leanoj_module.asyncio.create_task(leanoj_module.asyncio.sleep(60))
        coordinator._main_task = task  # type: ignore[attr-defined]

        original_wait_for = leanoj_module.asyncio.wait_for

        async def fake_wait_for(*_args, **_kwargs):
            coordinator._main_task = None  # type: ignore[attr-defined]
            raise leanoj_module.asyncio.TimeoutError

        leanoj_module.asyncio.wait_for = fake_wait_for  # type: ignore[assignment]
        try:
            await coordinator.stop()
        finally:
            leanoj_module.asyncio.wait_for = original_wait_for  # type: ignore[assignment]
            if not task.done():
                task.cancel()
                await leanoj_module.asyncio.gather(task, return_exceptions=True)

        self.assertTrue(task.cancelled())
        self.assertIsNone(coordinator._main_task)  # type: ignore[attr-defined]

    async def test_restore_latest_session_recovers_request_and_progress(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = LeanOJCoordinator()
                await coordinator.initialize(_request())
                coordinator._state.phase = "final_proof_loop"
                coordinator._state.selected_topic = "algebraic simplification"
                coordinator._state.final_attempt_count = 3
                coordinator._accepted_ideas.append("Use rfl after normalization.")
                coordinator._validated_topics.append("Template theorem shape")
                coordinator._final_attempts.append(
                    {
                        "request": "final Proof Solver solution",
                        "error_summary": "unknown tactic",
                        "lean_code": "bad",
                    }
                )
                await coordinator._persist_state()  # type: ignore[attr-defined]

                restored = LeanOJCoordinator()
                self.assertTrue(await restored.restore_latest_session(auto_resume=False))

                self.assertEqual(restored.get_state().phase, "final_proof_loop")
                self.assertFalse(restored.get_state().is_running)
                self.assertEqual(restored.get_state().final_attempt_count, 3)
                self.assertEqual(restored.get_status()["accepted_ideas"], ["Use rfl after normalization."])
                self.assertEqual(restored.get_status()["validated_topics"], ["Template theorem shape"])
                self.assertTrue(restored.get_status()["resume_available"])
                self.assertIsNotNone(restored._request)  # type: ignore[attr-defined]
                self.assertEqual(restored._request.lean_template, _request().lean_template)  # type: ignore[attr-defined]
            finally:
                system_config.data_dir = old_data_dir

    async def test_restore_latest_session_auto_resume_starts_interrupted_run(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = LeanOJCoordinator()
                await coordinator.initialize(_request())
                coordinator._state.phase = "path_decision"
                await coordinator._persist_state()  # type: ignore[attr-defined]

                restored = LeanOJCoordinator()
                called = []

                def fake_start_in_background() -> bool:
                    called.append(True)
                    return True

                restored.start_in_background = fake_start_in_background  # type: ignore[method-assign]

                self.assertTrue(await restored.restore_latest_session(auto_resume=True))
                self.assertEqual(called, [True])
            finally:
                system_config.data_dir = old_data_dir

    async def test_leanoj_context_allocation_direct_first(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                allocation = await leanoj_context_manager.allocate_context(
                    session_id="session_direct",
                    mode="final_solver",
                    user_prompt="Prove one equals one.",
                    lean_template="example : 1 = 1 := by\n  sorry",
                    task_request="Solve the final proof.",
                    context_window=131072,
                    max_output_tokens=25000,
                    accepted_ideas=["Use rfl after normalization."],
                    verified_subproofs=[],
                    partial_proofs=[],
                    failed_subproofs=[],
                    final_attempts=[],
                )

                self.assertIn("Use rfl after normalization.", allocation.direct_proof_context)
                self.assertEqual(allocation.rag_evidence_context, "")
                self.assertTrue(allocation.direct_sources)
            finally:
                system_config.data_dir = old_data_dir

    async def test_leanoj_context_rag_fallback_is_scoped(self) -> None:
        captured = {}

        async def fake_ensure(_source_name: str, _text: str) -> None:
            return None

        async def fake_retrieve(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="retrieved scoped LeanOJ evidence")

        old_ensure = leanoj_context_manager._ensure_source_indexed
        old_retrieve = leanoj_context_module.rag_manager.retrieve
        try:
            leanoj_context_manager._ensure_source_indexed = fake_ensure  # type: ignore[method-assign]
            leanoj_context_module.rag_manager.retrieve = fake_retrieve  # type: ignore[assignment]

            allocation = await leanoj_context_manager.allocate_context(
                session_id="session_rag",
                mode="brainstorm",
                user_prompt="Prove a theorem.",
                lean_template="example : True := by\n  sorry",
                task_request="Brainstorm proof ideas.",
                context_window=10000,
                max_output_tokens=1000,
                accepted_ideas=["large idea " * 4000],
                verified_subproofs=[],
                partial_proofs=[],
                failed_subproofs=[],
                final_attempts=[],
            )

            self.assertEqual(allocation.rag_evidence_context, "retrieved scoped LeanOJ evidence")
            self.assertEqual(captured["include_source_prefixes"], ["leanoj_session_rag_"])
            self.assertIn("leanoj_session_rag_accepted_ideas", captured["include_sources"])
            self.assertEqual(captured["exclude_sources"], None)
        finally:
            leanoj_context_manager._ensure_source_indexed = old_ensure  # type: ignore[method-assign]
            leanoj_context_module.rag_manager.retrieve = old_retrieve  # type: ignore[assignment]

    async def test_historical_final_cycle_packets_are_rag_only(self) -> None:
        captured = {}

        async def fake_ensure(_source_name: str, _text: str) -> None:
            return None

        async def fake_retrieve(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="historical final-cycle packet evidence")

        old_ensure = leanoj_context_manager._ensure_source_indexed
        old_retrieve = leanoj_context_module.rag_manager.retrieve
        try:
            leanoj_context_manager._ensure_source_indexed = fake_ensure  # type: ignore[method-assign]
            leanoj_context_module.rag_manager.retrieve = fake_retrieve  # type: ignore[assignment]

            allocation = await leanoj_context_manager.allocate_context(
                session_id="session_packets",
                mode="brainstorm",
                user_prompt="Prove a theorem.",
                lean_template="example : True := by\n  sorry",
                task_request="Generate brainstorm proof context.",
                context_window=131072,
                max_output_tokens=25000,
                accepted_ideas=[],
                verified_subproofs=[],
                partial_proofs=[],
                failed_subproofs=[],
                final_attempts=[],
                final_cycle_packets=[
                    {
                        "cycle_start_attempt": 1,
                        "cycle_end_attempt": 30,
                        "failed_attempt_count": 30,
                        "attempts": [{"request": "final", "error_summary": "failed", "lean_code": "bad"}],
                    }
                ],
            )

            self.assertNotIn("FINAL-CYCLE PACKET", allocation.direct_proof_context)
            self.assertEqual(allocation.rag_evidence_context, "historical final-cycle packet evidence")
            self.assertIn("leanoj_session_packets_final_cycle_packets", captured["include_sources"])
        finally:
            leanoj_context_manager._ensure_source_indexed = old_ensure  # type: ignore[method-assign]
            leanoj_context_module.rag_manager.retrieve = old_retrieve  # type: ignore[assignment]

    async def test_context_allocation_raises_when_useful_memory_would_drop(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "mandatory context overflow|could not preserve useful proof memory",
        ):
            await leanoj_context_manager.allocate_context(
                session_id="session_tiny",
                mode="brainstorm",
                user_prompt="user " * 2000,
                lean_template="template " * 2000,
                task_request="task " * 2000,
                context_window=2500,
                max_output_tokens=1000,
                accepted_ideas=["large idea " * 100],
                verified_subproofs=[],
                partial_proofs=[],
                failed_subproofs=[],
                final_attempts=[],
            )

    async def test_ensure_source_indexed_removes_stale_source_before_add(self) -> None:
        calls = []

        async def fake_remove(source_name: str) -> None:
            calls.append(("remove", source_name))

        async def fake_add(_text: str, source_name: str, **_kwargs) -> None:
            calls.append(("add", source_name))

        old_remove = leanoj_context_module.rag_manager.remove_document
        old_add = leanoj_context_module.rag_manager.add_text
        try:
            leanoj_context_module.rag_manager.remove_document = fake_remove  # type: ignore[assignment]
            leanoj_context_module.rag_manager.add_text = fake_add  # type: ignore[assignment]
            leanoj_context_manager._indexed_hashes.pop("leanoj_stale_source", None)

            await leanoj_context_manager._ensure_source_indexed("leanoj_stale_source", "proof memory")

            self.assertEqual(calls[0], ("remove", "leanoj_stale_source"))
            self.assertEqual(calls[1], ("add", "leanoj_stale_source"))
        finally:
            leanoj_context_module.rag_manager.remove_document = old_remove  # type: ignore[assignment]
            leanoj_context_module.rag_manager.add_text = old_add  # type: ignore[assignment]

    async def test_current_final_cycle_packet_persists_until_phase_clear(self) -> None:
        coordinator = await self._initialized_coordinator()
        request = _request()
        packet = {
            "cycle_start_attempt": 1,
            "cycle_end_attempt": 30,
            "failed_attempt_count": 30,
            "attempts": [{"request": "final", "error_summary": "failed", "lean_code": "bad"}],
        }
        coordinator._current_final_cycle_packet = packet  # type: ignore[attr-defined]

        async def fake_allocate_context(**kwargs):
            return LeanOJContextAllocation(
                current_final_cycle_packet="CURRENT FINAL-CYCLE FAILURE PACKET"
                if kwargs.get("current_final_cycle_packet")
                else ""
            )

        old_allocate = leanoj_module.leanoj_context_manager.allocate_context
        try:
            leanoj_module.leanoj_context_manager.allocate_context = fake_allocate_context  # type: ignore[assignment]

            first = await coordinator._build_context_blocks(  # type: ignore[attr-defined]
                request,
                request.topic_generator,
                mode="brainstorm",
                task_request="Generate brainstorm proof context.",
                include_current_final_cycle_packet=True,
            )
            second = await coordinator._build_context_blocks(  # type: ignore[attr-defined]
                request,
                request.topic_generator,
                mode="brainstorm",
                task_request="Generate brainstorm proof context.",
                include_current_final_cycle_packet=True,
            )
            coordinator._clear_current_final_cycle_packet()  # type: ignore[attr-defined]
            third = await coordinator._build_context_blocks(  # type: ignore[attr-defined]
                request,
                request.topic_generator,
                mode="brainstorm",
                task_request="Generate brainstorm proof context.",
                include_current_final_cycle_packet=True,
            )

            self.assertIn("CURRENT FINAL-CYCLE FAILURE PACKET", first["current_final_cycle_packet"])
            self.assertIn("CURRENT FINAL-CYCLE FAILURE PACKET", second["current_final_cycle_packet"])
            self.assertEqual(third["current_final_cycle_packet"], "")
        finally:
            leanoj_module.leanoj_context_manager.allocate_context = old_allocate  # type: ignore[assignment]

    async def test_final_solver_context_does_not_duplicate_working_proof_packet(self) -> None:
        coordinator = await self._initialized_coordinator()
        request = _request()
        await coordinator._write_master_proof(  # type: ignore[attr-defined]
            "import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
            summary="final solver draft",
        )
        await coordinator._set_current_working_proof_attempt(  # type: ignore[attr-defined]
            trigger="final_attempt_cycle_exhausted",
            requested_path="need_more_brainstorming",
            stuck_reason="Need more context.",
        )

        blocks = await coordinator._build_context_blocks(  # type: ignore[attr-defined]
            request,
            request.final_solver,
            mode="final_solver",
            task_request="Edit final proof.",
            include_current_final_cycle_packet=True,
        )

        self.assertEqual(blocks["current_working_proof_attempt"], "")
        self.assertNotIn("CURRENT WORKING PROOF ATTEMPT", "\n".join(blocks.values()))

    def test_final_cycle_packet_formats_partial_proofs(self) -> None:
        packet = {
            "cycle_start_attempt": 1,
            "cycle_end_attempt": 2,
            "failed_attempt_count": 2,
            "attempts": [{"request": "final", "error_summary": "failed", "lean_code": "bad"}],
            "partial_proofs": [
                {
                    "request": "final Proof Solver solution",
                    "target": "final",
                    "attempt": 2,
                    "placeholder_tokens": ["sorry"],
                    "summary": "Lean accepted the scaffold.",
                    "lean_code": "example : True := by\n  sorry",
                }
            ],
        }

        formatted = leanoj_context_module.LeanOJContextManager._format_final_cycle_packet(packet)

        self.assertIn("Partial final scaffolds captured during this cycle", formatted)
        self.assertIn("Lean accepted the scaffold", formatted)
        self.assertIn("example : True", formatted)

    async def test_initial_brainstorm_exit_prune_can_delete_accepted_idea_without_extending_phase(self) -> None:
        coordinator = await self._initialized_coordinator()
        request = _request()

        async def fake_call_json(_config, _task_prefix, role_id, prompt, **_kwargs):
            if role_id.startswith("leanoj_brainstorm_submitter"):
                return {"submission": "Redundant idea to prune.", "reasoning": "seed"}
            if role_id == "leanoj_brainstorm_prune_reviewer_1":
                return {
                    "action": "delete",
                    "idea_index": 1,
                    "new_content": "",
                    "reasoning": "It is redundant after review.",
                }
            if role_id == "leanoj_brainstorm_validator" and "PROPOSED OPERATION:" in prompt:
                return {"decision": "accept", "reasoning": "Deletion is safe."}
            if role_id == "leanoj_brainstorm_validator" and "SUBMISSIONS TO VALIDATE:" in prompt:
                submission_count = prompt.count("SUBMISSION ")
                return {
                    "decisions": [
                        {
                            "submission_number": index,
                            "decision": "accept",
                            "reasoning": "Accept seed.",
                            "summary": "accepted",
                        }
                        for index in range(1, submission_count + 1)
                    ]
                }
            if role_id == "leanoj_brainstorm_validator":
                return {"decision": "accept", "reasoning": "Accept seed.", "summary": "accepted"}
            raise AssertionError(f"Unexpected role {role_id}")

        coordinator._call_json = fake_call_json  # type: ignore[method-assign]

        await coordinator._brainstorm_until_path_check(  # type: ignore[attr-defined]
            request,
            phase_key="initial_brainstorm",
            max_accepts=1,
            sufficiency_interval=10,
            force_after_max=True,
        )

        self.assertEqual(coordinator._accepted_ideas, [])  # type: ignore[attr-defined]
        self.assertEqual(coordinator.get_state().brainstorm_acceptance_events, 1)
        self.assertEqual(coordinator.get_state().accepted_brainstorm_count, 0)
        self.assertEqual(coordinator.get_state().brainstorm_prune_reviews_performed, 1)
        self.assertEqual(coordinator.get_state().brainstorm_prune_operations_applied, 1)

    async def test_recursive_brainstorm_prune_rejection_leaves_ideas_unchanged(self) -> None:
        coordinator = await self._initialized_coordinator()
        request = _request()
        coordinator._accepted_ideas = ["Useful exact idea"]  # type: ignore[attr-defined]
        coordinator._accepted_idea_records = [  # type: ignore[attr-defined]
            {
                "content": "Useful exact idea",
                "submitter_index": 1,
                "phase": "recursive_brainstorm",
                "acceptance_event": 1,
            }
        ]
        coordinator.get_state().brainstorm_acceptance_events = 1

        async def fake_call_json(_config, _task_prefix, role_id, prompt, **_kwargs):
            if role_id == "leanoj_brainstorm_prune_reviewer_1":
                return {
                    "action": "edit",
                    "idea_index": 1,
                    "new_content": "Risky replacement",
                    "reasoning": "Maybe shorter.",
                }
            if role_id == "leanoj_brainstorm_validator":
                self.assertIn("PROPOSED OPERATION:", prompt)
                return {"decision": "reject", "reasoning": "Original still has unique value."}
            raise AssertionError(f"Unexpected role {role_id}")

        coordinator._call_json = fake_call_json  # type: ignore[method-assign]

        await coordinator._perform_brainstorm_prune_review(  # type: ignore[attr-defined]
            request,
            "recursive_brainstorm",
            reason="test recursive review",
        )

        self.assertEqual(coordinator._accepted_ideas, ["Useful exact idea"])  # type: ignore[attr-defined]
        self.assertEqual(coordinator.get_state().brainstorm_prune_operations_applied, 0)

    async def test_clear_all_removes_only_registered_leanoj_rag_sources(self) -> None:
        removed = []
        old_data_dir = system_config.data_dir

        async def fake_remove(source_name: str) -> None:
            removed.append(source_name)

        old_remove = leanoj_context_module.rag_manager.remove_document
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                leanoj_context_module.rag_manager.remove_document = fake_remove  # type: ignore[assignment]
                leanoj_context_manager._indexed_hashes["leanoj_registered_accepted_ideas"] = "hash"

                await leanoj_context_manager.clear_all()

                self.assertEqual(removed, ["leanoj_registered_accepted_ideas"])
            finally:
                system_config.data_dir = old_data_dir
                leanoj_context_module.rag_manager.remove_document = old_remove  # type: ignore[assignment]
                leanoj_context_manager._indexed_hashes.pop("leanoj_registered_accepted_ideas", None)

    def test_rag_source_scope_filter_limits_chunks_to_leanoj_session(self) -> None:
        chunks = [
            DocumentChunk(
                chunk_id="1",
                text="LeanOJ proof memory",
                source_file="leanoj_session_a_accepted_ideas",
                position=0,
                chunk_size=512,
            ),
            DocumentChunk(
                chunk_id="2",
                text="compiler paper memory",
                source_file="compiler_paper.txt",
                position=0,
                chunk_size=512,
            ),
        ]

        scoped = leanoj_context_module.rag_manager._filter_chunks_by_source_scope(
            chunks,
            include_source_prefixes=["leanoj_session_a_"],
        )

        self.assertEqual([chunk.source_file for chunk in scoped], ["leanoj_session_a_accepted_ideas"])

    async def test_final_cycle_packet_contains_exact_cycle_attempts(self) -> None:
        request = _request()
        request.final_attempts_per_cycle = 30
        old_data_dir = system_config.data_dir

        async def fake_call_json(*_args, **_kwargs):
            return {"lean_code": "import Mathlib\n\nexample : 1 = 1 := by\n  simp", "reasoning": "try"}

        class FakeLean:
            async def check_proof(self, _code: str, timeout: int = 120, **_kwargs):
                return SimpleNamespace(success=False, error_output="unsolved goals")

        old_get_lean4_client = leanoj_module.get_lean4_client
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = await self._initialized_coordinator()
                coordinator._call_json = fake_call_json  # type: ignore[method-assign]
                leanoj_module.get_lean4_client = lambda: FakeLean()  # type: ignore[assignment]

                await coordinator._final_proof_loop(request)

                packet = coordinator.get_status()["current_final_cycle_packet"]
                self.assertEqual(packet["failed_attempt_count"], 30)
                self.assertEqual(packet["cycle_start_attempt"], 1)
                self.assertEqual(packet["cycle_end_attempt"], 30)
                self.assertEqual(len(packet["attempts"]), 30)
            finally:
                leanoj_module.get_lean4_client = old_get_lean4_client  # type: ignore[assignment]
                system_config.data_dir = old_data_dir

    async def test_restore_reloads_full_final_attempt_artifacts(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = LeanOJCoordinator()
                await coordinator.initialize(_request())
                coordinator._state.phase = "final_proof_loop"
                coordinator._state.final_attempt_count = 25
                for index in range(25):
                    coordinator._final_attempts.append(
                        {
                            "request": "final Proof Solver solution",
                            "error_summary": f"error {index}",
                            "lean_code": f"bad {index}",
                        }
                    )
                await coordinator._persist_state()  # type: ignore[attr-defined]

                restored = LeanOJCoordinator()
                self.assertTrue(await restored.restore_latest_session(auto_resume=False))

                self.assertEqual(len(restored._final_attempts), 25)  # type: ignore[attr-defined]
                self.assertEqual(restored._final_attempts[0]["error_summary"], "error 0")  # type: ignore[attr-defined]
                self.assertEqual(len(restored.get_status()["final_attempts"]), 20)
            finally:
                system_config.data_dir = old_data_dir

    async def test_leanoj_artifact_sync_rewrites_same_length_edits(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                session_id = "same_length_edit"
                leanoj_context_manager._clear_sync_counts(session_id)  # type: ignore[attr-defined]
                await leanoj_context_manager.write_session_artifacts(
                    session_id=session_id,
                    accepted_ideas=["old idea"],
                    verified_subproofs=[],
                    partial_proofs=[],
                    failed_subproofs=[],
                    final_attempts=[],
                    final_cycle_packets=[],
                )
                await leanoj_context_manager.write_session_artifacts(
                    session_id=session_id,
                    accepted_ideas=["new idea"],
                    verified_subproofs=[],
                    partial_proofs=[],
                    failed_subproofs=[],
                    final_attempts=[],
                    final_cycle_packets=[],
                )

                artifacts = leanoj_context_manager.load_session_artifacts(session_id)
                self.assertEqual(artifacts["accepted_ideas"], ["new idea"])
            finally:
                leanoj_context_manager._clear_sync_counts("same_length_edit")  # type: ignore[attr-defined]
                system_config.data_dir = old_data_dir

    async def test_restore_reloads_master_proof_metadata_and_content(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = LeanOJCoordinator()
                await coordinator.initialize(_request())
                coordinator._state.phase = "final_proof_loop"  # type: ignore[attr-defined]
                await coordinator._write_master_proof(  # type: ignore[attr-defined]
                    "import Mathlib\n\nexample : 1 = 1 := by\n  rfl",
                    summary="resume test master proof",
                )
                await coordinator._persist_state()  # type: ignore[attr-defined]

                restored = LeanOJCoordinator()
                self.assertTrue(await restored.restore_latest_session(auto_resume=False))

                self.assertTrue(restored.get_state().master_proof_initialized)
                self.assertEqual(restored.get_state().master_proof_line_count, 4)
                self.assertIn("rfl", await restored._read_master_proof())  # type: ignore[attr-defined]
                self.assertEqual(restored._infer_resume_phase(), "final_proof_loop")  # type: ignore[attr-defined]
            finally:
                system_config.data_dir = old_data_dir

    async def test_clear_removes_leanoj_artifact_store(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                coordinator = LeanOJCoordinator()
                await coordinator.initialize(_request())
                coordinator._accepted_ideas.append("persisted idea")
                await coordinator._persist_state()  # type: ignore[attr-defined]

                self.assertTrue(leanoj_context_manager.artifacts_base_dir().exists())

                await coordinator.clear()

                self.assertFalse(leanoj_context_manager.artifacts_base_dir().exists())
                self.assertEqual(coordinator.get_status()["accepted_ideas"], [])
            finally:
                system_config.data_dir = old_data_dir


if __name__ == "__main__":
    unittest.main()
