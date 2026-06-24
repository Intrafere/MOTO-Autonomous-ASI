import unittest
from importlib import import_module
from types import SimpleNamespace

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.prompts.proof_prompts import build_proof_identification_prompt
from backend.shared.config import system_config
from backend.shared.models import ProofCandidate, ProofStageResult

coordinator_module = import_module("backend.autonomous.core.autonomous_coordinator")


class _FakeResearchMetadata:
    def __init__(self):
        self.checkpoint = None
        self.mark_calls = []
        self.clear_calls = []
        self.workflow_states = []

    async def save_proof_checkpoint(self, checkpoint):
        self.checkpoint = dict(checkpoint)

    async def get_proof_checkpoint(self, source_type=None, source_id=None, trigger=None):
        if not self.checkpoint:
            return None
        if source_type and self.checkpoint.get("source_type") != source_type:
            return None
        if source_id and self.checkpoint.get("source_id") != source_id:
            return None
        if trigger and self.checkpoint.get("trigger") != trigger:
            return None
        return dict(self.checkpoint)

    async def mark_proof_checkpoint_trigger_complete(
        self,
        source_type,
        source_id,
        trigger,
        source_title="",
    ):
        self.mark_calls.append(trigger)
        completed_triggers = set()
        if self.checkpoint:
            completed_triggers.update(self.checkpoint.get("completed_triggers") or [])
        completed_triggers.add(trigger)
        self.checkpoint = {
            **(self.checkpoint or {}),
            "source_type": source_type,
            "source_id": source_id,
            "source_title": source_title,
            "trigger": trigger,
            "status": "trigger_complete",
            "completed_triggers": sorted(completed_triggers),
        }

    async def clear_proof_checkpoint(self, source_type=None, source_id=None):
        self.clear_calls.append((source_type, source_id))
        self.checkpoint = None

    async def save_workflow_state(self, state):
        self.workflow_states.append(dict(state))


class _FakeProofDatabase:
    def __init__(self):
        self.inject_count = 0
        self.pending_retry_calls = []

    def inject_into_prompt(self, prompt):
        self.inject_count += 1
        return f"proof_context_{self.inject_count}::{prompt}"

    async def get_pending_retries(self, *args, **kwargs):
        self.pending_retry_calls.append((args, kwargs))
        return []


class _FakeBrainstormMemory:
    async def get_metadata(self, _topic_id):
        return SimpleNamespace(topic_prompt="Topic prompt", submission_count=0)

    async def get_database_content(self, _topic_id, *, strip_proofs=False):
        return "Brainstorm source."


class _FakeProofStage:
    def __init__(self, totals):
        self.totals = list(totals)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.totals) - 1)
        total = self.totals[index]
        return ProofStageResult(
            source_type=kwargs["source_type"],
            source_id=kwargs["source_id"],
            total_candidates=total,
            verified_count=total,
            novel_count=total,
        )


class _ErrorProofStage(_FakeProofStage):
    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return ProofStageResult(
            source_type=kwargs["source_type"],
            source_id=kwargs["source_id"],
            had_error=True,
            error_message="retryable proof stage error",
        )

def _configured_coordinator(fake_stage, *, allow_research_papers=True):
    coordinator = AutonomousCoordinator()
    coordinator._allow_mathematical_proofs = True
    coordinator._allow_research_papers = allow_research_papers
    coordinator._user_research_prompt = "Solve the user prompt."
    coordinator._writer_model = "proof-model"
    coordinator._writer_context = 4000
    coordinator._writer_max_tokens = 1000
    coordinator._high_param_model = "rigor-model"
    coordinator._high_param_context = 6000
    coordinator._high_param_max_tokens = 1500
    coordinator._validator_model = "validator-model"
    coordinator._validator_context = 4000
    coordinator._validator_max_tokens = 1000
    coordinator._proof_verification_stage = fake_stage
    return coordinator


class AutonomousAbstractExtractionTests(unittest.TestCase):
    def test_extract_abstract_from_plain_heading_after_attribution_header(self):
        coordinator = AutonomousCoordinator()

        abstract = coordinator._extract_abstract(
            "=" * 80
            + "\nAUTONOMOUS AI SOLUTION\n\n"
            + "Paper Title: Example\n"
            + "=" * 80
            + "\n\nAbstract\n\n"
            + "This is the real abstract paragraph. It should be stored, not the heading.\n\n"
            + "A second abstract paragraph should be preserved before the intro.\n\n"
            + "I. Introduction\n\nBody starts here."
        )

        self.assertIn("This is the real abstract paragraph", abstract)
        self.assertIn("A second abstract paragraph", abstract)
        self.assertNotEqual(abstract, "Abstract")
        self.assertNotIn("I. Introduction", abstract)

    def test_extract_abstract_from_markdown_heading(self):
        coordinator = AutonomousCoordinator()

        abstract = coordinator._extract_abstract(
            "# Abstract\n\nMarkdown abstract content.\n\n# Introduction\n\nIntro content."
        )

        self.assertEqual(abstract, "Markdown abstract content.")


class AutonomousProofRoundTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_lean4_enabled = system_config.lean4_enabled
        self.old_research_metadata = coordinator_module.research_metadata
        self.old_proof_database = coordinator_module.proof_database
        self.old_brainstorm_memory = coordinator_module.brainstorm_memory
        system_config.lean4_enabled = True
        coordinator_module.research_metadata = _FakeResearchMetadata()
        coordinator_module.proof_database = _FakeProofDatabase()
        coordinator_module.brainstorm_memory = _FakeBrainstormMemory()

    def tearDown(self):
        system_config.lean4_enabled = self.old_lean4_enabled
        coordinator_module.research_metadata = self.old_research_metadata
        coordinator_module.proof_database = self.old_proof_database
        coordinator_module.brainstorm_memory = self.old_brainstorm_memory

    def test_follow_up_prompt_uses_strict_question(self):
        prompt = build_proof_identification_prompt(
            user_prompt="Solve the problem.",
            source_type="paper",
            source_id="paper_001",
            source_content="Source content.",
            proof_round_index=2,
            proof_max_rounds=4,
            prior_round_results="Round 1: 1/1 candidates verified, 1 novel.",
        )

        self.assertIn(
            "Are there any proofs here to solve that directly solve the users prompt, "
            "or get us substantially closer to solving the users prompt.",
            prompt,
        )
        self.assertIn("Return TRUE only if the answer is yes.", prompt)
        self.assertIn("Round 1: 1/1 candidates verified, 1 novel.", prompt)

    async def test_explicit_lean_prompt_uses_standard_discovery(self):
        class FakeIdentificationAgent:
            def __init__(self):
                self.called = False
                self.kwargs = None

            async def identify_candidates(self, **kwargs):
                self.called = True
                self.kwargs = kwargs
                return True, [
                    ProofCandidate(
                        theorem_id="goal_direct_solution",
                        statement="A discovered theorem that aggressively solves the user prompt.",
                        expected_novelty_tier="mathematical_discovery",
                    ),
                    ProofCandidate(
                        theorem_id="goal_supporting_lemma",
                        statement="A discovered theorem that builds toward solving the user prompt.",
                        expected_novelty_tier="novel_variant",
                    ),
                ]

        fake_identifier = FakeIdentificationAgent()
        stage = ProofVerificationStage()

        candidates = await stage._resolve_candidates(
            theorem_candidates=None,
            identification_agent=fake_identifier,
            user_prompt=(
                "Solve the following Lean theorem.\n\n"
                "theorem direct_user_target {n : Nat} (hn : 0 < n) : n = n := by\n"
                "  ...\n\n"
                "----\n\n"
                "Helper Proof #1, verified:\n"
                "theorem helper_verified : True := by trivial\n"
            ),
            source_type="brainstorm",
            source_id="manual_aggregator",
            source_title="Manual Aggregator Database",
            content="Manual source.",
        )

        self.assertTrue(fake_identifier.called)
        self.assertEqual(fake_identifier.kwargs["user_research_prompt"].count("direct_user_target"), 1)
        self.assertEqual([candidate.theorem_id for candidate in candidates], [
            "goal_direct_solution",
            "goal_supporting_lemma",
        ])

    async def test_proofs_only_brainstorm_rounds_continue_until_no_candidates(self):
        fake_stage = _FakeProofStage([1, 1, 0, 1])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)

        await coordinator._run_proof_verification(
            content="Brainstorm source.",
            source_type="brainstorm",
            source_id="topic_001",
            source_title="Topic title",
        )

        self.assertEqual(len(fake_stage.calls), 3)
        self.assertEqual(
            [call["proof_round_index"] for call in fake_stage.calls],
            [1, 2, 3],
        )
        self.assertEqual(
            [call["trigger"] for call in fake_stage.calls],
            ["automatic", "automatic_round_2", "automatic_round_3"],
        )

    async def test_proofs_only_paper_rounds_cap_at_four(self):
        fake_stage = _FakeProofStage([1, 1, 1, 1, 1])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)

        await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_001",
            source_title="Paper title",
        )

        self.assertEqual(len(fake_stage.calls), 4)
        self.assertEqual(fake_stage.calls[-1]["trigger"], "automatic_round_4")
        self.assertTrue(all(call["proof_max_rounds"] == 4 for call in fake_stage.calls))

    async def test_papers_plus_proofs_automatic_checks_are_single_round(self):
        for source_type in ("brainstorm", "paper"):
            fake_stage = _FakeProofStage([1, 1, 1])
            coordinator = _configured_coordinator(fake_stage, allow_research_papers=True)

            await coordinator._run_proof_verification(
                content="Source.",
                source_type=source_type,
                source_id=f"{source_type}_001",
                source_title="Source title",
            )

            self.assertEqual(len(fake_stage.calls), 1)
            self.assertEqual(fake_stage.calls[0]["trigger"], "automatic")
            self.assertEqual(fake_stage.calls[0]["proof_round_index"], 1)
            self.assertEqual(fake_stage.calls[0]["proof_max_rounds"], 1)

    async def test_automatic_proof_check_uses_rigor_and_proofs_budget_for_all_sources(self):
        for source_type in ("brainstorm", "paper"):
            fake_stage = _FakeProofStage([0])
            coordinator = _configured_coordinator(fake_stage)

            await coordinator._run_proof_verification(
                content="Source.",
                source_type=source_type,
                source_id=f"{source_type}_001",
                source_title="Source title",
            )

            self.assertEqual(fake_stage.calls[0]["submitter_model"], "rigor-model")
            self.assertEqual(fake_stage.calls[0]["submitter_context"], 6000)
            self.assertEqual(fake_stage.calls[0]["submitter_max_tokens"], 1500)

    async def test_automatic_rounds_refresh_verified_proof_context(self):
        fake_stage = _FakeProofStage([1, 0])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)

        await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_001",
            source_title="Paper title",
        )

        self.assertEqual(len(fake_stage.calls), 2)
        self.assertTrue(fake_stage.calls[0]["user_prompt"].startswith("proof_context_1::"))
        self.assertTrue(fake_stage.calls[1]["user_prompt"].startswith("proof_context_2::"))

    async def test_later_round_resume_does_not_overwrite_active_checkpoint(self):
        resume_candidate = ProofCandidate(
            theorem_id="resume_target",
            statement="Resume theorem.",
            expected_novelty_tier="mathematical_discovery",
        )
        fake_metadata = coordinator_module.research_metadata
        fake_metadata.checkpoint = {
            "source_type": "paper",
            "source_id": "paper_resume",
            "source_title": "Paper title",
            "trigger": "automatic_round_2",
            "status": "running",
            "completed_triggers": ["automatic"],
            "candidates": [
                {
                    "index": 1,
                    "candidate": resume_candidate.model_dump(mode="json"),
                }
            ],
            "processed_candidate_ids": [],
            "attempts_by_candidate": {},
            "theorem_names_by_candidate": {},
            "results": [],
            "total_candidates": 1,
        }
        fake_stage = _FakeProofStage([0])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)

        await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_resume",
            source_title="Paper title",
        )

        self.assertEqual(len(fake_stage.calls), 1)
        self.assertEqual(fake_stage.calls[0]["trigger"], "automatic_round_2")
        self.assertEqual(
            fake_stage.calls[0]["theorem_candidates"][0].theorem_id,
            "resume_target",
        )
        self.assertNotEqual(fake_metadata.mark_calls[0], "automatic")

    async def test_automatic_follow_up_rounds_hold_source_reservation(self):
        class ReservationCheckingStage(_FakeProofStage):
            def __init__(self, totals):
                super().__init__(totals)
                self.reserved_during_calls = []

            async def run(self, **kwargs):
                self.reserved_during_calls.append(
                    await ProofVerificationStage.is_source_running(
                        kwargs["source_type"],
                        kwargs["source_id"],
                    )
                )
                return await super().run(**kwargs)

        fake_stage = ReservationCheckingStage([1, 0])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)

        await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_reserved",
            source_title="Paper title",
        )

        self.assertEqual(fake_stage.reserved_during_calls, [True, True])
        self.assertTrue(all(call["source_reserved"] for call in fake_stage.calls))
        self.assertTrue(all(not call["release_source_on_exit"] for call in fake_stage.calls))
        self.assertFalse(
            await ProofVerificationStage.is_source_running("paper", "paper_reserved")
        )

    async def test_proof_stage_error_returns_preserved_status(self):
        fake_stage = _ErrorProofStage([])
        coordinator = _configured_coordinator(fake_stage)

        status = await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_error",
            source_title="Paper title",
        )

        self.assertEqual(status, "error_preserved")
        self.assertEqual(len(fake_stage.calls), 1)

    async def test_brainstorm_proof_error_sets_stop_before_paper_handoff(self):
        fake_stage = _ErrorProofStage([])
        coordinator = _configured_coordinator(fake_stage)
        coordinator._current_topic_id = "topic_error"

        status = await coordinator._run_brainstorm_completion_proofs()

        self.assertEqual(status, "error_preserved")
        self.assertTrue(coordinator._stop_event.is_set())
        self.assertEqual(coordinator_module.research_metadata.clear_calls, [])

    async def test_proofs_only_handoff_saves_topic_boundary_without_paper_state(self):
        fake_stage = _FakeProofStage([0])
        coordinator = _configured_coordinator(fake_stage, allow_research_papers=False)
        coordinator._running = True
        coordinator._state.current_tier = "tier2_paper_writing"
        coordinator._current_topic_id = "topic_done"
        coordinator._current_paper_id = "paper_should_not_resume"
        coordinator._current_paper_title = "Should Not Resume"
        coordinator._current_reference_papers = ["paper_ref"]
        coordinator._current_reference_brainstorms = ["brainstorm_ref"]
        coordinator._brainstorm_paper_count = 2
        coordinator._current_brainstorm_paper_ids = ["paper_a"]
        coordinator._last_completed_paper_id = "paper_a"
        events = []

        async def capture_event(event, data):
            events.append((event, data))

        coordinator.set_broadcast_callback(capture_event)

        await coordinator._handle_papers_disabled_after_brainstorm()

        self.assertEqual(coordinator._state.current_tier, "tier1_aggregation")
        self.assertIsNone(coordinator._current_topic_id)
        self.assertIsNone(coordinator._current_paper_id)
        self.assertIsNone(coordinator._current_paper_title)
        self.assertEqual(coordinator._current_reference_papers, [])
        self.assertEqual(coordinator._current_reference_brainstorms, [])
        self.assertEqual(coordinator._brainstorm_paper_count, 0)
        self.assertEqual(coordinator._current_brainstorm_paper_ids, [])
        self.assertIsNone(coordinator._last_completed_paper_id)
        self.assertEqual(events[0][0], "research_papers_disabled_brainstorm_complete")
        self.assertEqual(
            coordinator_module.research_metadata.workflow_states[-1]["current_tier"],
            "tier1_aggregation",
        )
        self.assertEqual(
            coordinator_module.research_metadata.workflow_states[-1]["paper_phase"],
            "topic_exploration",
        )
        self.assertIsNone(
            coordinator_module.research_metadata.workflow_states[-1]["current_paper_id"]
        )

    async def test_paper_proof_error_does_not_clear_or_retry_checkpoint(self):
        fake_stage = _ErrorProofStage([])
        coordinator = _configured_coordinator(fake_stage)
        fake_metadata = coordinator_module.research_metadata
        fake_metadata.checkpoint = {
            "source_type": "paper",
            "source_id": "paper_error",
            "source_title": "Paper title",
            "trigger": "automatic",
            "status": "running",
            "completed_triggers": [],
            "candidates": [],
            "processed_candidate_ids": [],
            "attempts_by_candidate": {},
            "theorem_names_by_candidate": {},
            "results": [],
            "total_candidates": 1,
        }

        status = await coordinator._run_completed_paper_proof_checks(
            paper_id="paper_error",
            title="Paper title",
            content="Paper source.",
            source_brainstorm_ids=["topic_001"],
        )

        self.assertEqual(status, "stopped")
        self.assertTrue(coordinator._stop_event.is_set())
        self.assertEqual(fake_metadata.clear_calls, [])
        self.assertEqual(coordinator_module.proof_database.pending_retry_calls, [])

    async def test_completed_paper_proof_check_strips_existing_proof_appendix(self):
        fake_stage = _FakeProofStage([0])
        coordinator = _configured_coordinator(fake_stage)

        status = await coordinator._run_completed_paper_proof_checks(
            paper_id="paper_stripped",
            title="Paper title",
            content=(
                "Paper body sentinel.\n\n"
                "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
                "Lean 4 Code:\n"
                "theorem autonomous_paper_duplicate : True := by trivial\n"
            ),
            source_brainstorm_ids=[],
        )

        self.assertEqual(status, "complete")
        self.assertIn("Paper body sentinel.", fake_stage.calls[0]["content"])
        self.assertNotIn("autonomous_paper_duplicate", fake_stage.calls[0]["content"])

    async def test_retry_trigger_remains_single_round(self):
        fake_stage = _FakeProofStage([1, 1, 1])
        coordinator = _configured_coordinator(fake_stage)
        retry_candidate = ProofCandidate(
            theorem_id="retry_target",
            statement="Retry theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        await coordinator._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper_001",
            source_title="Paper title",
            theorem_candidates=[retry_candidate],
            trigger="retry",
        )

        self.assertEqual(len(fake_stage.calls), 1)
        self.assertEqual(fake_stage.calls[0]["trigger"], "retry")
        self.assertEqual(fake_stage.calls[0]["proof_max_rounds"], 1)

    async def test_manual_stage_wrapper_remains_single_round(self):
        class CaptureStage(ProofVerificationStage):
            def __init__(self):
                super().__init__()
                self.kwargs = None

            async def run(self, **kwargs):
                self.kwargs = kwargs
                return ProofStageResult(
                    source_type=kwargs["source_type"],
                    source_id=kwargs["source_id"],
                )

        stage = CaptureStage()
        await stage.run_manual(
            content="Manual source.",
            source_type="paper",
            source_id="paper_001",
            user_prompt="Prompt.",
            submitter_model="model",
            submitter_context=4000,
            submitter_max_tokens=1000,
            validator_model="validator",
            validator_context=4000,
            validator_max_tokens=1000,
            broadcast_fn=None,
            novel_proofs_db=object(),
        )

        self.assertEqual(stage.kwargs["trigger"], "manual")
        self.assertEqual(stage.kwargs.get("proof_max_rounds", 1), 1)


if __name__ == "__main__":
    unittest.main()
