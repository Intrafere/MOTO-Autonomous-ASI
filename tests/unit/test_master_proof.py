"""Master proof scaffolds: structure checks, solving, expansion, changes, budgets."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.autonomous.agents.master_proof_agent import (
    MasterProofAgent,
    MasterProofContextOverflow,
    MasterProofSubmission,
    ScaffoldChangeReview,
)
from backend.autonomous.agents.proof_identification_agent import ProofIdentificationAgent
from backend.autonomous.core import master_proof as master_proof_module
from backend.autonomous.core.master_proof import (
    MASTER_PROOF_ATTEMPTS_PER_OBLIGATION,
    MasterProofSolver,
    MasterProofState,
    MasterProofStore,
)
from backend.autonomous.core.proof_verification_stage import (
    _candidate_fingerprint,
    _master_proof_count,
)
from backend.shared.lean_scaffold import (
    open_obligations,
    parse_blocks,
    plan_obligation_replacement,
    resolve_theorem_name,
    scaffold_structure_problem,
)
from backend.shared.models import ProofAttemptFeedback, ProofCandidate

SCAFFOLD = """import Mathlib

/-- First step. -/
lemma step_one (n : Nat) : n + 0 = n := by
  sorry

@[simp]
lemma step_two (n : Nat) :
    0 + n = n := by
  sorry

def helper (n : Nat := 0) : Nat := n

theorem master (n : Nat) : n + 0 = n ∧ 0 + n = n := by
  exact ⟨step_one n, step_two n⟩
"""
STEP_ONE_SIG = "lemma step_one (n : Nat) : n + 0 = n"
STEP_ONE_PROOF = "lemma step_one (n : Nat) : n + 0 = n := by\n  simp"
STEP_TWO_PROOF = "lemma step_two (n : Nat) :\n    0 + n = n := by\n  simp"


NAMESPACED_SCAFFOLD = """import Mathlib

namespace MOTO.Demo

lemma step_one (n : Nat) : n + 0 = n := by
  sorry

theorem master (n : Nat) : n + 0 = n := by
  exact step_one n

end MOTO.Demo
"""


class LeanScaffoldStructureTests(unittest.TestCase):
    def test_namespace_qualified_master_name_resolves_to_its_declaration(self):
        self.assertEqual(resolve_theorem_name(NAMESPACED_SCAFFOLD, "MOTO.Demo.master"), "master")
        self.assertEqual(resolve_theorem_name(NAMESPACED_SCAFFOLD, "master"), "master")
        self.assertEqual(resolve_theorem_name(NAMESPACED_SCAFFOLD, "Other.missing"), "Other.missing")
        self.assertEqual(scaffold_structure_problem(NAMESPACED_SCAFFOLD, "master"), "")
        self.assertEqual([block.name for block in open_obligations(NAMESPACED_SCAFFOLD)], ["step_one"])

    def test_master_proof_count_counts_only_master_candidates(self):
        candidates = [
            ProofCandidate(theorem_id="a", statement="A", proposal_kind="master_proof"),
            ProofCandidate(theorem_id="b", statement="B"),
        ]
        self.assertEqual(_master_proof_count(candidates), 1)

    def test_blocks_keep_doc_comments_attributes_and_signatures(self):
        blocks = {block.name: block for block in parse_blocks(SCAFFOLD) if block.name}
        self.assertEqual(set(blocks), {"step_one", "step_two", "master"})
        self.assertIn("/-- First step. -/", blocks["step_one"].text)
        self.assertIn("@[simp]", blocks["step_two"].text)
        self.assertEqual(blocks["step_one"].signature, STEP_ONE_SIG)
        self.assertEqual(blocks["step_two"].signature, "lemma step_two (n : Nat) : 0 + n = n")
        self.assertFalse(blocks["master"].has_placeholder)
        self.assertEqual([block.name for block in open_obligations(SCAFFOLD)], ["step_one", "step_two"])

    def test_placeholder_in_comment_or_string_is_not_an_obligation(self):
        code = 'theorem done : True := by\n  -- sorry is not needed\n  trivial\n'
        self.assertEqual(open_obligations(code), [])

    def test_structure_rejects_missing_root_changed_root_and_stray_placeholder(self):
        self.assertEqual(scaffold_structure_problem(SCAFFOLD, "master"), "")
        self.assertIn("was not found", scaffold_structure_problem(SCAFFOLD, "absent"))
        self.assertIn(
            "was changed",
            scaffold_structure_problem(SCAFFOLD, "master", required_root_signature="theorem master : True"),
        )
        stray = SCAFFOLD.replace("def helper (n : Nat := 0) : Nat := n", "def helper : Nat := sorry")
        self.assertIn("every `sorry`", scaffold_structure_problem(stray, "master"))
        duplicate = SCAFFOLD + "\nlemma step_one : True := by\n  trivial\n"
        self.assertIn("unique", scaffold_structure_problem(duplicate, "master"))

    def test_master_theorem_must_stay_the_last_theorem(self):
        trailing = SCAFFOLD + "\nlemma after_root : True := by\n  trivial\n"
        self.assertIn("must be the last", scaffold_structure_problem(trailing, "master"))

    def test_lean_sorry_warnings_are_attributed_to_their_declaration(self):
        from backend.shared.lean_scaffold import declarations_with_sorry_warning

        diagnostics = (
            "MOTOProofCheck_x.lean:4:6: warning: declaration uses 'sorry'\n"
            "MOTOProofCheck_x.lean:8:6: warning: declaration uses 'sorry'\n"
            "MOTOProofCheck_x.lean:15:8: warning: unused variable `h`\n"
        )
        self.assertEqual(
            declarations_with_sorry_warning(SCAFFOLD, diagnostics), {"step_one", "step_two"}
        )
        self.assertEqual(declarations_with_sorry_warning(SCAFFOLD, ""), set())

    def test_replacement_closes_or_expands_one_obligation(self):
        closed = plan_obligation_replacement(SCAFFOLD, "step_one", STEP_ONE_SIG, STEP_ONE_PROOF)
        self.assertEqual(closed.problem, "")
        self.assertEqual(closed.new_obligations, ())
        self.assertEqual([block.name for block in open_obligations(closed.code)], ["step_two"])
        self.assertEqual(scaffold_structure_problem(closed.code, "master"), "")

        expanded = plan_obligation_replacement(
            SCAFFOLD,
            "step_one",
            STEP_ONE_SIG,
            "lemma aux (n : Nat) : n = n := by\n  sorry\n\n" + STEP_ONE_PROOF,
        )
        self.assertEqual([block.name for block in expanded.new_obligations], ["aux"])
        self.assertEqual(
            [block.name for block in open_obligations(expanded.code)], ["aux", "step_two"]
        )

    def test_replacement_rejects_contract_violations(self):
        def problem(replacement: str) -> str:
            return plan_obligation_replacement(SCAFFOLD, "step_one", STEP_ONE_SIG, replacement).problem

        self.assertIn("exactly once", problem("lemma other : True := by\n  trivial"))
        self.assertIn("was changed", problem("lemma step_one (n : Nat) : n = n := by\n  rfl"))
        self.assertIn("still contains", problem("lemma step_one (n : Nat) : n + 0 = n := by\n  sorry"))
        self.assertIn("import", problem("import Mathlib\n\n" + STEP_ONE_PROOF))
        self.assertIn(
            "must not repeat",
            problem("lemma step_two : True := by\n  trivial\n\n" + STEP_ONE_PROOF),
        )


class _FakeLean:
    """Accepts code unless it contains the marker ``BAD``; placeholder-free on request."""

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    async def check_proof(self, code, timeout=600, *, allow_placeholders=False):
        self.calls.append((code, allow_placeholders))
        ok = "BAD" not in code and (allow_placeholders or "sorry" not in code)
        return SimpleNamespace(
            success=ok,
            error_output="" if ok else "error: rejected",
            goal_states="",
            checked_source=code,
            compile_receipt=None,
        )


class _ScriptedAgent:
    """Returns scripted submissions per obligation; records feedback it was shown."""

    def __init__(self, scaffold=None, obligations=None):
        self.scaffold = list(scaffold or [])
        self.obligations = {name: list(items) for name, items in (obligations or {}).items()}
        self.seen_attempt_counts: dict[str, list[int]] = {}

    async def propose_scaffold(self, **kwargs):
        return self.scaffold.pop(0)

    async def attempt_obligation(self, *, obligation_name, prior_attempts, **kwargs):
        self.seen_attempt_counts.setdefault(obligation_name, []).append(len(prior_attempts))
        await asyncio.sleep(0)
        script = self.obligations[obligation_name]
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, Exception):
            raise item
        return item


class _Validator:
    def __init__(self, review):
        self.review_result = review
        self.calls = 0

    async def review(self, **kwargs):
        self.calls += 1
        return self.review_result


def _candidate() -> ProofCandidate:
    return ProofCandidate(theorem_id="thm_1", statement="Master statement.", proposal_kind="master_proof")


def _scaffold_submission(code=SCAFFOLD) -> MasterProofSubmission:
    return MasterProofSubmission(lean_code=code, root_theorem_name="master")


def _proof(code: str) -> MasterProofSubmission:
    return MasterProofSubmission(lean_code=code, action="submit_proof")


class MasterProofSolverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.lean = _FakeLean()
        self.events: list[tuple[str, dict]] = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name) / "master_proofs"
        patches = [
            patch.object(master_proof_module, "get_lean4_client", lambda: self.lean),
            patch.object(master_proof_module, "require_lean_workspace", lambda *args: None),
            patch.object(master_proof_module, "raise_for_lean_workspace_failure", lambda result: None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def _solver(self, agent, *, validator=None, state=None, should_stop=None, max_parallel=0):
        async def emit(kind, payload):
            self.events.append((kind, payload))

        self.state = state or MasterProofState(
            master_key="key", theorem_id="thm_1", statement="Master statement."
        )
        return MasterProofSolver(
            state=self.state,
            store=MasterProofStore(self.directory),
            agent=agent,
            change_validator=validator or _Validator(ScaffoldChangeReview(decided=True, approved=True)),
            candidate=_candidate(),
            user_prompt="prompt",
            source_type="brainstorm",
            source_title="",
            source_content="source",
            emit=emit,
            should_stop=should_stop,
            max_parallel=max_parallel,
        )

    async def test_namespaced_scaffold_with_qualified_root_name_is_accepted_first_try(self):
        agent = _ScriptedAgent(
            scaffold=[
                MasterProofSubmission(
                    lean_code=NAMESPACED_SCAFFOLD, root_theorem_name="MOTO.Demo.master"
                )
            ],
            obligations={"step_one": [_proof(STEP_ONE_PROOF)]},
        )
        outcome = await self._solver(agent).run()

        self.assertTrue(outcome.success)
        self.assertEqual(self.state.root_name, "master")
        self.assertEqual(len(self.state.scaffold_attempts), 0)
        accepted = next(p for kind, p in self.events if kind == "scaffold_accepted")
        self.assertIn("1-step master proof", accepted["message"])
        self.assertIn("end MOTO.Demo", outcome.lean_code)

    async def test_only_failures_in_a_row_count_toward_the_budget(self):
        bad = _proof("lemma step_one (n : Nat) : n + 0 = n := by\n  exact BAD")
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [bad, bad, _proof(STEP_ONE_PROOF)],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        outcome = await self._solver(agent).run()

        self.assertTrue(outcome.success)
        # Two failures were recorded; the closing success added nothing to the budget.
        self.assertEqual(len(self.state.nodes["step_one"].attempts), 2)
        self.assertEqual(len(self.state.nodes["step_two"].attempts), 0)
        failed = [p["message"] for kind, p in self.events if kind == "attempt_failed"]
        self.assertIn("2 of 15 failures in a row", failed[-1])
        started = [p["message"] for kind, p in self.events if kind == "attempt_started"]
        self.assertTrue(any("0 of 15 failures in a row so far" in item for item in started))

    def _kinds(self) -> list[str]:
        return [kind for kind, _payload in self.events]

    async def test_scaffold_then_parallel_sorries_close_and_final_check_is_placeholder_free(self):
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={"step_one": [_proof(STEP_ONE_PROOF)], "step_two": [_proof(STEP_TWO_PROOF)]},
        )
        outcome = await self._solver(agent).run()

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.theorem_name, "master")
        self.assertNotIn("sorry", outcome.lean_code)
        self.assertTrue(outcome.final_feedback.success)
        self.assertEqual(self.lean.calls[-1][1], False)
        self.assertTrue(all(allow for _code, allow in self.lean.calls[:-1]))
        self.assertEqual(self.state.status, "verified")
        self.assertIn("scaffold_accepted", self._kinds())
        self.assertEqual(self._kinds().count("sorry_closed"), 2)
        self.assertEqual(self._kinds()[-1], "completed")
        self.assertTrue((self.directory / "master_proof_key.lean").is_file())

    async def test_sorry_can_expand_repeatedly_and_children_get_fresh_budgets(self):
        expand_one = _proof(
            "lemma aux_a (n : Nat) : n = n := by\n  sorry\n\n" + STEP_ONE_PROOF
        )
        expand_aux = _proof(
            "lemma aux_b (n : Nat) : n = n := by\n  sorry\n\nlemma aux_a (n : Nat) : n = n := by\n  exact aux_b n"
        )
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [expand_one],
                "step_two": [_proof(STEP_TWO_PROOF)],
                "aux_a": [_proof("lemma aux_a (n : Nat) : n = n := by\n  BAD"), expand_aux],
                "aux_b": [_proof("lemma aux_b (n : Nat) : n = n := by\n  rfl")],
            },
        )
        outcome = await self._solver(agent).run()

        self.assertTrue(outcome.success)
        self.assertEqual(self._kinds().count("sorry_expanded"), 2)
        self.assertEqual(self.state.nodes["aux_a"].depth, 1)
        self.assertEqual(self.state.nodes["aux_b"].depth, 2)
        self.assertEqual(self.state.nodes["aux_b"].parent, "aux_a")
        # aux_a failed once and then saw exactly that one feedback entry.
        self.assertEqual(agent.seen_attempt_counts["aux_a"], [0, 1])
        self.assertEqual(agent.seen_attempt_counts["aux_b"], [0])

    async def test_proof_that_lean_reports_as_using_sorry_does_not_close_its_obligation(self):
        hidden = "lemma step_one (n : Nat) : n + 0 = n := by\n  exact HIDDEN_PLACEHOLDER"
        lean = self.lean

        async def check_proof(code, timeout=600, *, allow_placeholders=False):
            lean.calls.append((code, allow_placeholders))
            line = next(
                (index for index, text in enumerate(code.split("\n"), start=1) if "HIDDEN_PLACEHOLDER" in text),
                0,
            )
            return SimpleNamespace(
                success=True,
                error_output="",
                goal_states="",
                checked_source=code,
                compile_receipt=None,
                diagnostic_output=(
                    f"Check.lean:{line - 1}:6: warning: declaration uses 'sorry'\n" if line else ""
                ),
            )

        lean.check_proof = check_proof
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [_proof(hidden), _proof(STEP_ONE_PROOF)],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        outcome = await self._solver(agent).run()

        self.assertTrue(outcome.success)
        self.assertNotIn("HIDDEN_PLACEHOLDER", outcome.lean_code)
        rejected = self.state.nodes["step_one"].attempts[0]
        self.assertIn("still uses", rejected.error_output)
        self.assertEqual(agent.seen_attempt_counts["step_one"], [0, 1])

    async def test_scaffold_change_keeps_finished_obligations_on_record(self):
        changed = SCAFFOLD.replace(
            "lemma step_one (n : Nat) : n + 0 = n := by\n  sorry", STEP_ONE_PROOF
        ).replace(
            "lemma step_two (n : Nat) :\n    0 + n = n := by\n  sorry",
            "lemma step_two (n : Nat) (h : True) :\n    0 + n = n := by\n  sorry",
        ).replace("step_two n⟩", "step_two n trivial⟩")

        class Agent(_ScriptedAgent):
            async def attempt_obligation(self, *, obligation_name, obligation_signature, **kwargs):
                if obligation_name == "step_one":
                    return _proof(STEP_ONE_PROOF)
                if "(h : True)" in obligation_signature:
                    return _proof("lemma step_two (n : Nat) (h : True) :\n    0 + n = n := by\n  simp")
                if "simp" not in kwargs["scaffold_code"].split("lemma step_two")[0]:
                    await asyncio.sleep(0)
                    return _proof("lemma step_two (n : Nat) :\n    0 + n = n := by\n  BAD")
                return MasterProofSubmission(lean_code=changed, action="change_scaffold")

        outcome = await asyncio.wait_for(
            self._solver(Agent(scaffold=[_scaffold_submission()])).run(), 5
        )

        self.assertTrue(outcome.success)
        self.assertIn("scaffold_changed", self._kinds())
        self.assertEqual(self.state.nodes["step_one"].status, "closed")

    async def test_one_sorry_failing_its_budget_fails_the_whole_master_proof(self):
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [_proof("lemma step_one (n : Nat) : n + 0 = n := by\n  BAD")],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        outcome = await self._solver(agent).run()

        self.assertFalse(outcome.success)
        self.assertEqual(self.state.status, "failed")
        self.assertEqual(
            len(self.state.nodes["step_one"].attempts), MASTER_PROOF_ATTEMPTS_PER_OBLIGATION
        )
        self.assertEqual(
            agent.seen_attempt_counts["step_one"],
            list(range(MASTER_PROOF_ATTEMPTS_PER_OBLIGATION)),
        )
        self.assertIn("step_one", outcome.final_feedback.error_output)
        self.assertIn("MASTER PROOF FAILED", outcome.final_feedback.error_output)
        self.assertEqual(self._kinds()[-1], "failed")
        # The partial scaffold with the closed sibling stays on disk.
        saved = (self.directory / "master_proof_key.lean").read_text(encoding="utf-8")
        self.assertIn("simp", saved)
        self.assertIn("sorry", saved)

    async def test_scaffold_budget_exhaustion_fails_without_solving(self):
        agent = _ScriptedAgent(scaffold=[_scaffold_submission(SCAFFOLD + "\n-- BAD\n")] * 20)
        outcome = await self._solver(agent).run()

        self.assertFalse(outcome.success)
        self.assertEqual(
            len(self.state.scaffold_attempts), MASTER_PROOF_ATTEMPTS_PER_OBLIGATION
        )
        self.assertEqual(self.state.nodes, {})

    async def test_validator_approved_change_replaces_scaffold_and_resets_affected_sorries(self):
        changed = SCAFFOLD.replace(
            "lemma step_two (n : Nat) :\n    0 + n = n := by\n  sorry",
            "lemma step_two (n : Nat) (h : True) :\n    0 + n = n := by\n  sorry",
        ).replace("step_two n⟩", "step_two n trivial⟩")
        new_step_two = "lemma step_two (n : Nat) (h : True) :\n    0 + n = n := by\n  simp"
        change_proposed = {"done": False}

        class Agent(_ScriptedAgent):
            async def attempt_obligation(self, *, obligation_name, obligation_signature, **kwargs):
                if obligation_name == "step_one":
                    if not change_proposed["done"]:
                        change_proposed["done"] = True
                        return MasterProofSubmission(lean_code=changed, action="change_scaffold")
                    return _proof(STEP_ONE_PROOF)
                if "(h : True)" not in obligation_signature:
                    await asyncio.sleep(3600)  # stale worker; must be reset, not awaited
                return _proof(new_step_two)

        validator = _Validator(ScaffoldChangeReview(decided=True, approved=True, reasoning="needed"))
        agent = Agent(scaffold=[_scaffold_submission()])
        outcome = await asyncio.wait_for(self._solver(agent, validator=validator).run(), 5)

        self.assertTrue(outcome.success)
        self.assertEqual(validator.calls, 1)
        changed_events = [payload for kind, payload in self.events if kind == "scaffold_changed"]
        self.assertEqual(len(changed_events), 1)
        self.assertEqual(changed_events[0]["reset_obligations"], ["step_two"])
        self.assertIn("(h : True)", outcome.lean_code)
        # The unchanged proposing obligation kept its history: the accepted change is one attempt.
        self.assertEqual(len(self.state.nodes["step_one"].attempts), 1)
        self.assertEqual(self.state.nodes["step_two"].attempts, [])

    async def test_validator_rejection_keeps_scaffold_and_counts_as_feedback(self):
        changed = SCAFFOLD.replace("-- First step.", "-- changed").replace(
            "/-- First step. -/", "/-- Changed. -/"
        )
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [
                    MasterProofSubmission(lean_code=changed, action="change_scaffold"),
                    _proof(STEP_ONE_PROOF),
                ],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        validator = _Validator(ScaffoldChangeReview(decided=True, approved=False, reasoning="no gain"))
        outcome = await self._solver(agent, validator=validator).run()

        self.assertTrue(outcome.success)
        self.assertIn("scaffold_change_rejected", self._kinds())
        self.assertNotIn("scaffold_changed", self._kinds())
        self.assertNotIn("Changed.", outcome.lean_code)
        rejection = self.state.nodes["step_one"].attempts[0]
        self.assertIn("REJECTED BY VALIDATOR: no gain", rejection.error_output)
        self.assertEqual(agent.seen_attempt_counts["step_one"], [0, 1])

    async def test_change_may_not_alter_the_master_theorem_statement(self):
        weakened = SCAFFOLD.replace(
            "theorem master (n : Nat) : n + 0 = n ∧ 0 + n = n", "theorem master (n : Nat) : True"
        )
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [
                    MasterProofSubmission(lean_code=weakened, action="change_scaffold"),
                    _proof(STEP_ONE_PROOF),
                ],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        validator = _Validator(ScaffoldChangeReview(decided=True, approved=True))
        outcome = await self._solver(agent, validator=validator).run()

        self.assertTrue(outcome.success)
        self.assertEqual(validator.calls, 0)
        self.assertIn("was changed", self.state.nodes["step_one"].attempts[0].error_output)

    async def test_saved_state_resumes_open_sorries_with_their_feedback(self):
        failing = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={
                "step_one": [
                    _proof("lemma step_one (n : Nat) : n + 0 = n := by\n  BAD"),
                    RuntimeError("provider down"),
                ],
                "step_two": [_proof(STEP_TWO_PROOF)],
            },
        )
        with self.assertRaises(RuntimeError):
            await self._solver(failing, max_parallel=1).run()

        restored = await MasterProofStore(self.directory).load("key")
        self.assertEqual(restored.status, "solving")
        self.assertEqual(len(restored.nodes["step_one"].attempts), 1)
        resumed = _ScriptedAgent(
            obligations={"step_one": [_proof(STEP_ONE_PROOF)], "step_two": [_proof(STEP_TWO_PROOF)]}
        )
        outcome = await self._solver(resumed, state=restored).run()

        self.assertTrue(outcome.success)
        self.assertEqual(resumed.seen_attempt_counts["step_one"], [1])

    async def test_stop_cancels_workers(self):
        stop = {"requested": False}

        class Agent(_ScriptedAgent):
            async def attempt_obligation(self, **kwargs):
                stop["requested"] = True
                await asyncio.sleep(0)
                return _proof(STEP_ONE_PROOF)

        with self.assertRaises(asyncio.CancelledError):
            await self._solver(
                Agent(scaffold=[_scaffold_submission()]), should_stop=lambda: stop["requested"]
            ).run()
        self.assertEqual(self.state.status, "solving")

    async def test_context_overflow_sets_the_master_proof_aside(self):
        overflow = MasterProofContextOverflow(
            ProofAttemptFeedback(attempt=1, theorem_id="thm_1", error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW: x")
        )
        agent = _ScriptedAgent(
            scaffold=[_scaffold_submission()],
            obligations={"step_one": [overflow], "step_two": [_proof(STEP_TWO_PROOF)]},
        )
        outcome = await self._solver(agent).run()

        self.assertFalse(outcome.success)
        self.assertIsNotNone(outcome.overflow_feedback)
        # The scaffold call reached the model, so the stage must not treat this as
        # a round in which nothing could run.
        self.assertIsNotNone(outcome.reached_feedback)
        self.assertFalse(
            master_proof_module.ProofAttemptFeedback.model_validate(
                outcome.reached_feedback.model_dump()
            ).overflow_origin
        )
        self.assertEqual(self.state.status, "solving")


def _failed(attempt: int, size: int = 400) -> ProofAttemptFeedback:
    return ProofAttemptFeedback(
        attempt=attempt,
        theorem_id="thm_1",
        reasoning=f"reason {attempt}",
        lean_code=f"-- attempt {attempt}\n" + ("x " * size),
        error_output=f"error: failure {attempt}",
        failure_kind="lean_rejected",
        lean_was_run=True,
    )


class MasterProofFeedbackWindowTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, attempts, context_window, user_prompt="prompt"):
        agent = MasterProofAgent("model", context_window, 100, "role")
        prompts: list[str] = []

        async def fake_completion(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return {
                "choices": [
                    {"message": {"content": '{"action": "submit_proof", "lean_code": "lemma a : True := by trivial", "reasoning": "r"}'}}
                ]
            }

        with patch(
            "backend.autonomous.agents.master_proof_agent.api_client_manager.generate_completion",
            side_effect=fake_completion,
        ):
            result = await agent.attempt_obligation(
                user_prompt=user_prompt,
                source_type="brainstorm",
                source_title="",
                candidate=_candidate(),
                source_content="source",
                root_theorem_name="master",
                scaffold_code=SCAFFOLD,
                obligation_name="step_one",
                obligation_signature=STEP_ONE_SIG,
                prior_attempts=attempts,
                attempt_number=len(attempts) + 1,
            )
        return result, prompts

    async def test_verified_proof_library_is_left_out_before_any_feedback_is_dropped(self):
        from backend.autonomous.prompts.proof_prompts import (
            VERIFIED_PROOF_LIBRARY_END,
            VERIFIED_PROOF_LIBRARY_START,
        )
        from backend.shared.utils import count_tokens

        library = (
            f"{VERIFIED_PROOF_LIBRARY_START}\n"
            + "\n".join(f"PROOF {index}: library_marker lemma_{index}" for index in range(900))
            + f"\n{VERIFIED_PROOF_LIBRARY_END}"
        )
        with_library = f"{library}\n\nprompt"
        attempts = [_failed(index, size=5) for index in range(1, 4)]

        # Roomy window: the library stays.
        _result, roomy = await self._run(attempts, context_window=200_000, user_prompt=with_library)
        self.assertIn("library_marker", roomy[0])

        # Window that fits everything except the library: the library goes, and
        # every feedback entry, the scaffold, and the source stay.
        _result, bare = await self._run(attempts, context_window=200_000)
        window = count_tokens(bare[0]) + 100 + 500 + 200
        self.assertLess(window, count_tokens(roomy[0]))
        result, prompts = await self._run(attempts, context_window=window, user_prompt=with_library)

        self.assertIsInstance(result, MasterProofSubmission)
        self.assertNotIn("library_marker", prompts[0])
        self.assertIn("[No verified proof library context injected.]", prompts[0])
        for index in range(1, 4):
            self.assertIn(f"error: failure {index}", prompts[0])
        self.assertIn("theorem master", prompts[0])

    async def test_all_fifteen_rejections_are_shown_when_they_fit(self):
        attempts = [_failed(index, size=5) for index in range(1, 16)]
        result, prompts = await self._run(attempts, context_window=200_000)

        self.assertIsInstance(result, MasterProofSubmission)
        for index in range(1, 16):
            self.assertIn(f"error: failure {index}", prompts[0])

    async def test_oldest_rejections_drop_first_and_the_newest_always_stays(self):
        attempts = [_failed(index) for index in range(1, 16)]
        _result, full = await self._run(attempts, context_window=200_000)
        from backend.shared.utils import count_tokens

        one_entry = count_tokens(full[0]) // 15
        window = count_tokens(full[0]) - 6 * one_entry + 100 + 500 + 50
        result, prompts = await self._run(attempts, context_window=window)

        self.assertIsInstance(result, MasterProofSubmission)
        self.assertNotIn("error: failure 1\n", prompts[0])
        self.assertIn("error: failure 15", prompts[0])
        shown = [index for index in range(1, 16) if f"ATTEMPT {index}:" in prompts[0]]
        self.assertEqual(shown, list(range(shown[0], 16)))
        self.assertGreater(shown[0], 1)

    async def test_newest_only_overflow_raises_instead_of_dropping_the_newest(self):
        attempts = [_failed(index) for index in range(1, 4)]
        with self.assertRaises(MasterProofContextOverflow) as raised:
            await self._run(attempts, context_window=700)
        self.assertEqual(raised.exception.feedback.overflow_origin, "local_preflight")


class MasterProofProposalContractTests(unittest.TestCase):
    @staticmethod
    def _payload(kind_field: str) -> str:
        return (
            '{"has_provable_theorems": true, "theorems": [{"theorem_id": "thm_1", '
            '"statement": "S", "expected_novelty_tier": "mathematical_discovery", '
            '"prompt_relevance_rationale": "a", "novelty_rationale": "b", '
            f'"why_not_standard_known_result": "c"{kind_field}}}]}}'
        )

    def test_proposal_kind_defaults_to_individual_and_accepts_master_proof(self):
        _has, default = ProofIdentificationAgent._parse_candidate_payload(self._payload(""))
        _has, master = ProofIdentificationAgent._parse_candidate_payload(
            self._payload(', "proposal_kind": "master_proof"')
        )
        _has, unknown = ProofIdentificationAgent._parse_candidate_payload(
            self._payload(', "proposal_kind": "something_else"')
        )
        self.assertEqual(default[0].proposal_kind, "individual")
        self.assertEqual(master[0].proposal_kind, "master_proof")
        self.assertEqual(unknown[0].proposal_kind, "individual")

    def test_individual_candidate_fingerprint_is_unchanged_by_the_new_field(self):
        individual = ProofCandidate(theorem_id="thm_1", statement="S")
        master = individual.model_copy(update={"proposal_kind": "master_proof"})
        self.assertNotEqual(
            _candidate_fingerprint(individual, "brainstorm", "b1"),
            _candidate_fingerprint(master, "brainstorm", "b1"),
        )


class MasterProofListReviewPromptTests(unittest.TestCase):
    def test_list_validator_reviews_a_master_proof_as_a_whole(self):
        from backend.autonomous.prompts.proof_prompts import (
            build_proof_candidate_list_validation_prompt,
        )

        prompt = build_proof_candidate_list_validation_prompt(
            user_prompt="prompt",
            source_type="brainstorm",
            source_id="b1",
            source_title="",
            candidates_json=_candidate().model_dump_json(),
        )
        self.assertIn('"proposal_kind":"master_proof"', prompt)
        self.assertIn("Review it holistically", prompt)
        self.assertIn("never reject a master proof because steps along its route", prompt)


class MasterProofStageRoutingTests(unittest.IsolatedAsyncioTestCase):
    """The proof stage sends master proof candidates through the scaffold solver."""

    async def _run_stage(self, outcome, *, novel_proofs_db=None):
        from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
        from backend.autonomous.core import proof_verification_stage as stage_module

        captured = {}

        class FakeSolver:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def run(self):
                await captured["emit"]("scaffold_accepted", {"message": "accepted"})
                return outcome

        stage = stage_module.ProofVerificationStage()
        events: list[tuple[str, dict]] = []

        async def broadcast(event, data):
            events.append((event, data))

        async def unexpected(*args, **kwargs):
            raise AssertionError("individual proof attempts must not run for a master proof")

        async def prepared(**kwargs):
            return kwargs["theorem_candidate"]

        with patch.object(stage_module, "MasterProofSolver", FakeSolver), patch.object(
            stage_module, "require_lean_workspace", lambda *args: None
        ), patch.object(stage, "_prepare_candidate", prepared), patch.object(
            stage, "_run_smt_check", unexpected
        ), patch.object(ProofFormalizationAgent, "prove_candidate", unexpected), patch.object(
            ProofFormalizationAgent, "prove_candidate_tactic_script", unexpected
        ):
            result = await stage._run_lean_pipeline_for_candidate(
                theorem_candidate=_candidate(),
                base_event={"source_type": "brainstorm", "source_id": "b1"},
                proof_label="A",
                user_prompt="prompt",
                source_type="brainstorm",
                source_id="b1",
                source_content="source",
                source_title="",
                submitter_model="solver/model",
                submitter_context=64_000,
                submitter_max_tokens=4_000,
                role_suffix="brainstorm",
                trigger="automatic",
                novel_proofs_db=novel_proofs_db,
                broadcast_fn=broadcast,
                run_id="run-1",
                validator_model="validator/model",
                validator_context=32_000,
                validator_max_tokens=2_000,
            )
        return result, events, captured

    async def test_verified_master_proof_becomes_the_candidate_outcome(self):
        from backend.autonomous.core.master_proof import MasterProofOutcome

        final = ProofAttemptFeedback(
            attempt=7,
            theorem_id="thm_1",
            lean_code="theorem master : True := trivial",
            success=True,
            lean_was_run=True,
        )
        result, events, captured = await self._run_stage(
            MasterProofOutcome(
                success=True, theorem_name="master", lean_code=final.lean_code, final_feedback=final
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.theorem_name, "master")
        self.assertEqual(result.lean_code, final.lean_code)
        self.assertIs(result.attempts[-1], final)
        names = [event for event, _data in events]
        self.assertIn("proof_lean_accepted", names)
        progress = next(data for event, data in events if event == "master_proof_progress")
        self.assertEqual(progress["kind"], "scaffold_accepted")
        self.assertEqual(progress["proof_label"], "A")
        self.assertEqual(captured["agent"].role_id, "autonomous_proof_formalization_brainstorm")
        self.assertEqual(captured["agent"].model_id, "solver/model")
        self.assertEqual(
            captured["change_validator"].role_id, "autonomous_proof_novelty_brainstorm"
        )
        self.assertEqual(captured["change_validator"].model_id, "validator/model")

    async def test_failed_master_proof_is_one_failed_candidate_without_individual_fallback(self):
        from backend.autonomous.core.master_proof import MasterProofOutcome

        failure = ProofAttemptFeedback(
            attempt=20,
            theorem_id="thm_1",
            lean_code=SCAFFOLD,
            error_output="MASTER PROOF FAILED: one obligation was not closed within 15 attempts.",
            failure_kind="lean_rejected",
            lean_was_run=True,
        )
        result, events, _captured = await self._run_stage(
            MasterProofOutcome(success=False, lean_code=SCAFFOLD, final_feedback=failure)
        )

        self.assertFalse(result.success)
        self.assertIs(result.attempts[-1], failure)
        names = [event for event, _data in events]
        self.assertIn("proof_attempt_failed", names)
        self.assertNotIn("proof_attempts_exhausted", names)

    async def test_master_proof_overflow_uses_the_candidate_overflow_path(self):
        from backend.autonomous.core.master_proof import MasterProofOutcome

        overflow = ProofAttemptFeedback(
            attempt=1,
            theorem_id="thm_1",
            error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW: too large",
            overflow_origin="local_preflight",
            prompt_tokens=90_000,
            max_input_tokens=60_000,
        )
        result, events, _captured = await self._run_stage(
            MasterProofOutcome(success=False, overflow_feedback=overflow)
        )

        self.assertFalse(result.success)
        names = [event for event, _data in events]
        self.assertIn("proof_context_overflow", names)
        self.assertNotIn("proof_attempt_failed", names)
        self.assertNotIn("proof_attempts_exhausted", names)
        self.assertEqual(len(result.attempts), 1)
        # The round result needs the overflow details to explain a blocked round.
        self.assertEqual(result.context_overflow_payload["prompt_tokens"], 90_000)
        self.assertEqual(result.context_overflow_payload["overflow_origin"], "local_preflight")

    async def test_try_to_prove_discovery_may_mix_individual_and_master_proofs(self):
        """A manual proof check keeps each discovered candidate's proposal kind into Phase A."""
        from unittest.mock import AsyncMock

        from backend.autonomous.core import proof_verification_stage as stage_module
        from backend.shared.config import system_config
        from backend.shared.models import (
            ProofCandidateListValidation,
            ProofCandidateNoveltyDecision,
        )

        discovered = [
            ProofCandidate(theorem_id="thm_1", statement="Individual statement."),
            ProofCandidate(
                theorem_id="thm_2", statement="Master statement.", proposal_kind="master_proof"
            ),
        ]
        validation = ProofCandidateListValidation(
            results=[
                ProofCandidateNoveltyDecision(
                    theorem_id=candidate.theorem_id, decision="approve_novel", reasoning="novel"
                )
                for candidate in discovered
            ],
            feedback="suitable",
        )
        reviewed: list[list[str]] = []
        phase_a: dict[str, str] = {}
        stopped = {"value": False}

        async def validate(self, *, candidates, **kwargs):
            reviewed.append([candidate.proposal_kind for candidate in candidates])
            return validation

        async def pipeline(self, *, theorem_candidate, proof_label, **kwargs):
            phase_a[theorem_candidate.theorem_id] = theorem_candidate.proposal_kind
            if len(phase_a) == len(discovered):
                stopped["value"] = True
            return stage_module._LeanVerificationOutcome(
                candidate=theorem_candidate,
                proof_label=proof_label,
                success=False,
                theorem_name="",
                lean_code="",
                attempts=[],
            )

        async def broadcast(event, data):
            return None

        stage = stage_module.ProofVerificationStage()
        with patch.object(system_config, "lean4_enabled", True), patch.object(
            stage_module.ProofIdentificationAgent,
            "identify_candidates",
            new=AsyncMock(return_value=(True, discovered)),
        ), patch.object(
            stage_module.ProofCandidateListValidator, "validate", new=validate
        ), patch.object(
            stage_module.ProofVerificationStage, "_run_lean_pipeline_for_candidate", new=pipeline
        ), patch.object(
            stage_module.ProofVerificationStage,
            "_load_open_proof_targets",
            new=AsyncMock(return_value=([], "")),
        ):
            await stage.run_manual(
                content="Mandatory source.",
                source_type="paper",
                source_id="paper-1",
                source_title="Paper",
                user_prompt="Solve the objective.",
                submitter_model="submitter",
                submitter_context=8000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=object(),
                source_reserved=True,
                release_source_on_exit=False,
                source_reservation_token="reservation",
                should_stop=lambda: stopped["value"],
                run_id="run-1",
            )

        self.assertEqual(reviewed, [["individual", "master_proof"]])
        self.assertEqual(phase_a, {"thm_1": "individual", "thm_2": "master_proof"})

    async def test_failed_master_state_restarts_only_for_a_fresh_proposal(self):
        from backend.autonomous.core.master_proof import MasterProofOutcome, master_proof_key

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            key = master_proof_key(
                run_id="run-1",
                source_type="brainstorm",
                source_id="b1",
                statement="Master statement.",
            )
            await MasterProofStore(directory).save(
                MasterProofState(
                    master_key=key,
                    theorem_id="thm_1",
                    statement="Master statement.",
                    status="failed",
                    scaffold_code=SCAFFOLD,
                    failure_reason="old failure",
                )
            )
            database = SimpleNamespace(get_master_proofs_dir=lambda: directory)
            _result, _events, captured = await self._run_stage(
                MasterProofOutcome(success=False), novel_proofs_db=database
            )
        self.assertEqual(captured["state"].status, "scaffolding")
        self.assertEqual(captured["state"].scaffold_code, "")


    async def test_overflow_after_reaching_the_model_keeps_the_real_attempt_on_record(self):
        from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
        from backend.autonomous.core.master_proof import MasterProofOutcome

        reached = ProofAttemptFeedback(
            attempt=1,
            theorem_id="thm_1",
            error_output="SCAFFOLD REJECTED: example",
            failure_kind="integrity_rejected",
        )
        overflow = ProofAttemptFeedback(
            attempt=2,
            theorem_id="thm_1",
            error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW: too large",
            overflow_origin="local_preflight",
            prompt_tokens=90_000,
            max_input_tokens=80_000,
        )
        result, _events, _captured = await self._run_stage(
            MasterProofOutcome(success=False, overflow_feedback=overflow, reached_feedback=reached)
        )

        self.assertEqual([attempt.attempt for attempt in result.attempts], [1, 2])
        self.assertFalse(ProofFormalizationAgent.is_context_overflow_feedback(result.attempts[0]))
        self.assertTrue(ProofFormalizationAgent.is_context_overflow_feedback(result.attempts[-1]))


if __name__ == "__main__":
    unittest.main()
