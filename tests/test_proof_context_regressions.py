import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.api.routes import compiler as compiler_route
from backend.api.routes import proofs as proofs_route
from backend.autonomous.agents import proof_formalization_agent as proof_formalization_module
from backend.autonomous.agents import reference_selector as reference_selector_module
from backend.autonomous.agents.final_answer import certainty_assessor as certainty_assessor_module
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.autonomous.prompts.proof_prompts import build_proof_formalization_prompt
from backend.shared.config import system_config
from backend.shared.models import ProofCandidate


class _FakePaperLibrary:
    async def get_metadata(self, _paper_id):
        return None

    async def get_history_paper(self, session_id, paper_id):
        return {
            "content": "History paper full content sentinel.",
            "title": f"History title {session_id}:{paper_id}",
            "user_prompt": "History user prompt sentinel.",
        }

    @staticmethod
    def strip_verified_proofs_from_content(content):
        marker = "=== PROOFS ATTACHED TO THIS PAPER"
        return content.split(marker, 1)[0].rstrip()


class _FakeResearchMetadata:
    async def get_user_prompt(self):
        return "Current session prompt should not be used."

    async def get_base_user_prompt(self):
        return "Base prompt should not be used."


class _FakeActiveProofDatabase:
    def inject_into_prompt(self, prompt):
        return f"ACTIVE_DB::{prompt}"


class ProofContextRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_formalization_prompt_separates_verified_proof_context(self):
        injected_prompt = (
            "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ===\n"
            "PROOF 1 [Mathematical Discovery]: Existing theorem.\n"
            "Lean 4 Code:\n"
            "theorem existing_verified : True := by trivial\n"
            "---\n"
            "=== END VERIFIED PROOFS ===\n\n"
            "Solve the actual user prompt."
        )

        prompt = build_proof_formalization_prompt(
            user_prompt=injected_prompt,
            source_type="paper",
            theorem_statement="Target theorem.",
            formal_sketch="Use the source.",
            full_source_content="Paper body without proof appendix.",
            source_excerpt="Local target excerpt.",
            prior_attempts=[],
        )

        user_prompt_section = prompt.split(
            "USER RESEARCH PROMPT:",
            1,
        )[1].split("VERIFIED PROOF LIBRARY CONTEXT", 1)[0]
        proof_context_section = prompt.split(
            "VERIFIED PROOF LIBRARY CONTEXT",
            1,
        )[1].split("SOURCE TYPE:", 1)[0]

        self.assertIn("Solve the actual user prompt.", user_prompt_section)
        self.assertNotIn("existing_verified", user_prompt_section)
        self.assertIn("existing_verified", proof_context_section)

    async def test_legacy_history_manual_check_uses_legacy_proof_database(self):
        old_paper_library = proofs_route.paper_library
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        old_data_dir = proofs_route.system_config.data_dir
        proofs_route.paper_library = _FakePaperLibrary()
        proofs_route.research_metadata = _FakeResearchMetadata()
        proofs_route.proof_database = _FakeActiveProofDatabase()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                proofs_dir = Path(tmpdir) / "proofs"
                proofs_dir.mkdir(parents=True)
                (proofs_dir / "proofs_index.json").write_text(
                    json.dumps(
                        {
                            "next_proof_id": 2,
                            "proofs": [
                                {
                                    "proof_id": "proof_001",
                                    "theorem_statement": "Legacy verified theorem sentinel.",
                                    "source_type": "paper",
                                    "source_id": "paper_007",
                                    "lean_code": "theorem legacy_verified : True := by trivial",
                                    "novel": True,
                                    "novelty_tier": "mathematical_discovery",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                proofs_route.system_config.data_dir = tmpdir

                _content, _title, user_prompt = await proofs_route._resolve_manual_source(
                    SimpleNamespace(source_type="paper", source_id="legacy:paper_007")
                )
        finally:
            proofs_route.paper_library = old_paper_library
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database
            proofs_route.system_config.data_dir = old_data_dir

        self.assertIn("Legacy verified theorem sentinel", user_prompt)
        self.assertIn("theorem legacy_verified", user_prompt)
        self.assertNotIn("ACTIVE_DB::", user_prompt)

    async def test_history_manual_check_strips_appended_paper_proofs_from_source(self):
        class FakeHistoryPaperLibrary(_FakePaperLibrary):
            async def get_history_paper(self, _session_id, _paper_id):
                return {
                    "content": (
                        "History paper body sentinel.\n\n"
                        "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
                        "Lean 4 proof:\n"
                        "theorem appended_duplicate : True := by trivial\n"
                    ),
                    "title": "History title",
                    "user_prompt": "History prompt",
                }

        old_paper_library = proofs_route.paper_library
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        try:
            proofs_route.paper_library = FakeHistoryPaperLibrary()
            proofs_route.research_metadata = _FakeResearchMetadata()
            proofs_route.proof_database = _FakeActiveProofDatabase()

            content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                SimpleNamespace(source_type="paper", source_id="legacy:paper_007")
            )
        finally:
            proofs_route.paper_library = old_paper_library
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database

        self.assertIn("History paper body sentinel.", content)
        self.assertNotIn("appended_duplicate", content)

    async def test_manual_brainstorm_source_read_requests_stripped_proofs(self):
        class FakeBrainstormMemory:
            def __init__(self):
                self.strip_proofs = None

            async def get_metadata(self, _topic_id):
                return SimpleNamespace(topic_prompt="Topic prompt")

            async def get_database_content(self, _topic_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                return "Brainstorm body sentinel."

        fake_brainstorm_memory = FakeBrainstormMemory()
        old_brainstorm_memory = proofs_route.brainstorm_memory
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        try:
            proofs_route.brainstorm_memory = fake_brainstorm_memory
            proofs_route.research_metadata = _FakeResearchMetadata()
            proofs_route.proof_database = _FakeActiveProofDatabase()

            content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                SimpleNamespace(source_type="brainstorm", source_id="topic_001")
            )
        finally:
            proofs_route.brainstorm_memory = old_brainstorm_memory
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database

        self.assertTrue(fake_brainstorm_memory.strip_proofs)
        self.assertEqual(content, "Brainstorm body sentinel.")

    async def test_manual_current_paper_source_read_requests_stripped_proofs(self):
        class FakeCurrentPaperLibrary(_FakePaperLibrary):
            def __init__(self):
                self.strip_proofs = None

            async def get_metadata(self, _paper_id):
                return SimpleNamespace(title="Current paper", source_brainstorm_ids=[])

            async def get_paper_content(self, _paper_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                return "Current paper body sentinel."

        fake_paper_library = FakeCurrentPaperLibrary()
        old_paper_library = proofs_route.paper_library
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        try:
            proofs_route.paper_library = fake_paper_library
            proofs_route.research_metadata = _FakeResearchMetadata()
            proofs_route.proof_database = _FakeActiveProofDatabase()

            content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                SimpleNamespace(source_type="paper", source_id="paper_001")
            )
        finally:
            proofs_route.paper_library = old_paper_library
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database

        self.assertTrue(fake_paper_library.strip_proofs)
        self.assertIn("Current paper body sentinel.", content)

    async def test_manual_current_paper_includes_full_source_brainstorm_without_character_cap(self):
        long_middle = "M" * 60000
        full_brainstorm = (
            "Brainstorm head sentinel.\n"
            f"{long_middle}\n"
            "Brainstorm tail sentinel beyond old cap.\n\n"
            "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
            "theorem stripped_brainstorm_duplicate : True := by trivial\n"
        )

        class FakeCurrentPaperLibrary(_FakePaperLibrary):
            async def get_metadata(self, _paper_id):
                return SimpleNamespace(title="Current paper", source_brainstorm_ids=["topic_big"])

            async def get_paper_content(self, _paper_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                content = (
                    "Current paper body sentinel.\n\n"
                    "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
                    "theorem stripped_paper_duplicate : True := by trivial\n"
                )
                if strip_proofs:
                    return content.split("=== PROOFS ATTACHED TO THIS PAPER", 1)[0].rstrip()
                return content

        class FakeBrainstormMemory:
            def __init__(self):
                self.strip_proofs = None

            async def get_database_content(self, _topic_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                if not strip_proofs:
                    return full_brainstorm
                return full_brainstorm.split("=== PROOFS GENERATED FROM THIS BRAINSTORM", 1)[0].rstrip()

        fake_paper_library = FakeCurrentPaperLibrary()
        fake_brainstorm_memory = FakeBrainstormMemory()
        old_paper_library = proofs_route.paper_library
        old_brainstorm_memory = proofs_route.brainstorm_memory
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        try:
            proofs_route.paper_library = fake_paper_library
            proofs_route.brainstorm_memory = fake_brainstorm_memory
            proofs_route.research_metadata = _FakeResearchMetadata()
            proofs_route.proof_database = _FakeActiveProofDatabase()

            content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                SimpleNamespace(source_type="paper", source_id="paper_big")
            )
        finally:
            proofs_route.paper_library = old_paper_library
            proofs_route.brainstorm_memory = old_brainstorm_memory
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database

        self.assertTrue(fake_paper_library.strip_proofs)
        self.assertTrue(fake_brainstorm_memory.strip_proofs)
        self.assertIn("PAPER CONTENT:", content)
        self.assertIn("Current paper body sentinel.", content)
        self.assertIn("SOURCE BRAINSTORM topic_big:", content)
        self.assertIn("Brainstorm head sentinel.", content)
        self.assertIn("Brainstorm tail sentinel beyond old cap.", content)
        self.assertNotIn("source context truncated", content)
        self.assertNotIn("stripped_paper_duplicate", content)
        self.assertNotIn("stripped_brainstorm_duplicate", content)

    async def test_history_manual_check_includes_same_session_source_brainstorm(self):
        class FakeHistoryPaperLibrary(_FakePaperLibrary):
            async def get_history_paper(self, _session_id, _paper_id):
                return {
                    "content": "History paper body sentinel.",
                    "title": "History title",
                    "user_prompt": "History prompt",
                    "source_brainstorm_ids": ["topic_009"],
                }

        old_paper_library = proofs_route.paper_library
        old_research_metadata = proofs_route.research_metadata
        old_proof_database = proofs_route.proof_database
        old_auto_sessions_base_dir = proofs_route.system_config.auto_sessions_base_dir
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sessions_dir = Path(tmpdir) / "auto_sessions"
                brainstorms_dir = sessions_dir / "session_001" / "brainstorms"
                brainstorms_dir.mkdir(parents=True)
                (brainstorms_dir / "brainstorm_topic_009.txt").write_text(
                    "History brainstorm body sentinel.\n\n"
                    "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
                    "theorem history_brainstorm_duplicate : True := by trivial\n",
                    encoding="utf-8",
                )

                proofs_route.system_config.auto_sessions_base_dir = str(sessions_dir)
                proofs_route.paper_library = FakeHistoryPaperLibrary()
                proofs_route.research_metadata = _FakeResearchMetadata()
                proofs_route.proof_database = _FakeActiveProofDatabase()

                content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                    SimpleNamespace(source_type="paper", source_id="session_001:paper_007")
                )
        finally:
            proofs_route.paper_library = old_paper_library
            proofs_route.research_metadata = old_research_metadata
            proofs_route.proof_database = old_proof_database
            proofs_route.system_config.auto_sessions_base_dir = old_auto_sessions_base_dir

        self.assertIn("PAPER CONTENT:", content)
        self.assertIn("History paper body sentinel.", content)
        self.assertIn("SOURCE BRAINSTORM topic_009:", content)
        self.assertIn("History brainstorm body sentinel.", content)
        self.assertNotIn("history_brainstorm_duplicate", content)

    def test_verified_proof_library_injection_is_marker_deduped(self):
        db = ProofDatabase()
        db._index_data = {
            "next_proof_id": 2,
            "proofs": [
                {
                    "proof_id": "proof_001",
                    "theorem_statement": "Stored theorem sentinel.",
                    "source_type": "paper",
                    "source_id": "paper_001",
                    "lean_code": "theorem stored_proof_sentinel : True := by trivial",
                    "novel": True,
                    "novelty_tier": "mathematical_discovery",
                }
            ],
        }

        once = db.inject_into_prompt("User prompt sentinel.")
        twice = db.inject_into_prompt(once)

        self.assertEqual(once, twice)
        self.assertEqual(
            twice.count("=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ==="),
            1,
        )
        self.assertIn("stored_proof_sentinel", twice)

    async def test_saved_compiler_proof_content_strips_appended_paper_proofs(self):
        old_read_manual_aggregator_context = compiler_route._read_manual_aggregator_context

        async def no_aggregator_context():
            return ""

        compiler_route._read_manual_aggregator_context = no_aggregator_context
        try:
            content = await compiler_route._build_saved_compiler_proof_content(
                "Manual compiler paper body sentinel.\n\n"
                "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
                "Lean 4 proof:\n"
                "theorem compiler_appended_duplicate : True := by trivial\n"
            )
        finally:
            compiler_route._read_manual_aggregator_context = old_read_manual_aggregator_context

        self.assertIn("Manual compiler paper body sentinel.", content)
        self.assertNotIn("compiler_appended_duplicate", content)

    def test_paper_proof_stripping_preserves_self_review_after_fallback_proofs(self):
        stripped = PaperLibrary.strip_verified_proofs_from_content(
            "Manual compiler paper body sentinel.\n\n"
            "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
            "Lean 4 proof:\n"
            "theorem fallback_duplicate : True := by trivial\n\n"
            "AI Self-Review and Limitations\n"
            "Self-review content sentinel.\n"
        )

        self.assertIn("Manual compiler paper body sentinel.", stripped)
        self.assertNotIn("fallback_duplicate", stripped)
        self.assertIn("AI Self-Review and Limitations", stripped)
        self.assertIn("Self-review content sentinel.", stripped)

    async def test_reference_selector_expanded_papers_request_stripped_proofs(self):
        class FakePaperLibrary:
            def __init__(self):
                self.strip_proofs = None

            async def get_paper_content(self, _paper_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                return "Reference paper body sentinel."

            async def get_outline(self, _paper_id):
                return "Reference outline sentinel."

        fake_paper_library = FakePaperLibrary()
        old_paper_library = reference_selector_module.paper_library
        try:
            reference_selector_module.paper_library = fake_paper_library
            selector = reference_selector_module.ReferenceSelectorAgent(
                model_id="model",
                context_window=4000,
                max_output_tokens=1000,
            )

            expanded = await selector._get_expanded_papers(
                ["paper_001"],
                [{"paper_id": "paper_001", "title": "Reference paper"}],
            )
        finally:
            reference_selector_module.paper_library = old_paper_library

        self.assertTrue(fake_paper_library.strip_proofs)
        self.assertEqual(expanded[0]["content"], "Reference paper body sentinel.")

    async def test_certainty_assessor_expanded_papers_request_stripped_proofs(self):
        class FakePaperLibrary:
            def __init__(self):
                self.strip_proofs = None

            async def get_paper_content(self, _paper_id, *, strip_proofs=False):
                self.strip_proofs = strip_proofs
                return "Tier 3 certainty paper body sentinel."

            async def get_outline(self, _paper_id):
                return "Tier 3 certainty outline sentinel."

        fake_paper_library = FakePaperLibrary()
        old_paper_library = certainty_assessor_module.paper_library
        try:
            certainty_assessor_module.paper_library = fake_paper_library
            assessor = certainty_assessor_module.CertaintyAssessor(
                submitter_model="model",
                validator_model="validator",
                context_window=4000,
                max_output_tokens=1000,
            )

            expanded = await assessor._get_expanded_papers(
                ["paper_001"],
                [{"paper_id": "paper_001", "title": "Tier 3 paper"}],
            )
        finally:
            certainty_assessor_module.paper_library = old_paper_library

        self.assertTrue(fake_paper_library.strip_proofs)
        self.assertEqual(expanded[0]["content"], "Tier 3 certainty paper body sentinel.")

    async def test_brainstorm_submissions_list_strips_appended_proofs(self):
        memory = BrainstormMemory()
        with tempfile.TemporaryDirectory() as tmpdir:
            memory._base_dir = Path(tmpdir)
            db_path = memory._get_database_path("topic_001")
            db_path.write_text(
                "Brainstorm topic header\n"
                "================================================================================\n"
                "SUBMISSION #1 | Accepted: 2026-05-22T00:00:00\n"
                "================================================================================\n\n"
                "Accepted brainstorm content sentinel.\n\n"
                "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
                "Proof 1: Appended proof should not be parsed as submission content\n"
                "Lean 4 Code:\n"
                "theorem brainstorm_appended_duplicate : True := by trivial\n",
                encoding="utf-8",
            )

            submissions = await memory.get_submissions_list("topic_001")

        self.assertEqual(len(submissions), 1)
        self.assertIn("Accepted brainstorm content sentinel.", submissions[0]["content"])
        self.assertNotIn("brainstorm_appended_duplicate", submissions[0]["content"])

    async def test_full_source_context_overflow_is_retryable_stage_error(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        record_failed_calls = 0
        candidate = ProofCandidate(
            theorem_id="overflow",
            statement="Oversized source theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def fake_prepare_candidate(**kwargs):
            return kwargs["theorem_candidate"]

        async def fake_smt_check(**_kwargs):
            return None

        class FakeProofDb:
            async def record_failed_candidate(self, *_args, **_kwargs):
                nonlocal record_failed_calls
                record_failed_calls += 1

        stage._prepare_candidate = fake_prepare_candidate
        stage._run_smt_check = fake_smt_check
        try:
            result = await stage.run(
                content="OVERSIZED FULL SOURCE SENTINEL " * 500,
                source_type="brainstorm",
                source_id="topic_overflow",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=250,
                submitter_max_tokens=50,
                validator_model="validator",
                validator_context=1000,
                validator_max_tokens=100,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                theorem_candidates=[candidate],
                append_to_source=False,
            )
        finally:
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("MANDATORY FULL SOURCE CONTEXT OVERFLOW", result.error_message)
        self.assertEqual(record_failed_calls, 0)
        self.assertNotIn("proof_attempts_exhausted", events)

    async def test_codex_max_output_incomplete_does_not_crash_proof_stage(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []
        candidate = ProofCandidate(
            theorem_id="codex_incomplete",
            statement="Codex incomplete theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def fake_prepare_candidate(**kwargs):
            return kwargs["theorem_candidate"]

        async def fake_smt_check(**_kwargs):
            return None

        async def raise_codex_incomplete(*_args, **_kwargs):
            raise RuntimeError(
                "OpenAI Codex failed for role 'autonomous_proof_formalization_paper' "
                "and no LM Studio fallback is configured: OpenAI Codex completion failed: "
                '{"type":"response.incomplete","response":{"status":"incomplete",'
                '"incomplete_details":{"reason":"max_output_tokens"}}}'
            )

        class FakeProofDb:
            pass

        old_generate_completion = proof_formalization_module.api_client_manager.generate_completion
        proof_formalization_module.api_client_manager.generate_completion = raise_codex_incomplete
        stage._prepare_candidate = fake_prepare_candidate
        stage._run_smt_check = fake_smt_check
        try:
            result = await stage.run(
                content="paper source",
                source_type="paper",
                source_id="paper_001",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                theorem_candidates=[candidate],
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_formalization_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("MODEL OUTPUT INCOMPLETE", result.error_message)
        self.assertNotIn("response.incomplete", result.error_message)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_attempts_exhausted", events)


if __name__ == "__main__":
    unittest.main()
