import json
import importlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from backend.api.routes import aggregator as aggregator_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import proofs as proofs_route
from backend.autonomous.agents import proof_formalization_agent as proof_formalization_module
from backend.autonomous.agents import proof_identification_agent as proof_identification_module
from backend.autonomous.agents import reference_selector as reference_selector_module
from backend.autonomous.agents.final_answer import certainty_assessor as certainty_assessor_module
from backend.autonomous.core import proof_novelty as proof_novelty_module
from backend.autonomous.core import proof_registration as proof_registration_module
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.autonomous.prompts.proof_prompts import build_proof_formalization_prompt
from backend.aggregator.memory.shared_training import (
    clear_manual_aggregator_prompt,
    load_manual_aggregator_prompt,
    save_manual_aggregator_prompt,
)
from backend.compiler.memory.manual_prompt import (
    clear_manual_compiler_prompt,
    load_manual_compiler_prompt,
    save_manual_compiler_prompt,
)
from backend.shared.config import system_config
from backend.shared.api_client_manager import RetryableProviderError
from backend.shared.models import (
    ProofCandidate,
    ProofCheckRequest,
    ProofRecord,
    ProofRuntimeConfigSnapshot,
    ProofRoleConfigSnapshot,
    SubmitterConfig,
)


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
    async def test_manual_aggregator_runtime_prefers_active_manual_config_over_stale_request_snapshot(self):
        previous_state = {
            "submitter_configs": proofs_route.coordinator.submitter_configs,
            "validator_model": proofs_route.coordinator.validator_model,
            "validator_provider": proofs_route.coordinator.validator_provider,
            "validator_openrouter_provider": proofs_route.coordinator.validator_openrouter_provider,
            "validator_openrouter_reasoning_effort": proofs_route.coordinator.validator_openrouter_reasoning_effort,
            "validator_lm_studio_fallback": proofs_route.coordinator.validator_lm_studio_fallback,
            "validator_context_window": proofs_route.coordinator.validator_context_window,
            "validator_max_tokens": proofs_route.coordinator.validator_max_tokens,
            "validator_supercharge_enabled": proofs_route.coordinator.validator_supercharge_enabled,
        }
        stale_request_snapshot = ProofRuntimeConfigSnapshot(
            brainstorm=ProofRoleConfigSnapshot(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                context_window=400000,
                max_output_tokens=128000,
            ),
            paper=ProofRoleConfigSnapshot(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                context_window=400000,
                max_output_tokens=128000,
            ),
            validator=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="~google/gemini-flash-latest",
                context_window=1048576,
                max_output_tokens=65536,
            ),
        ).model_dump(mode="json")

        try:
            proofs_route.coordinator.submitter_configs = [
                SubmitterConfig(
                    submitter_id=1,
                    provider="openrouter",
                    model_id="anthropic/claude-opus-4.7",
                    openrouter_provider="Anthropic",
                    context_window=1000000,
                    max_output_tokens=128000,
                )
            ]
            proofs_route.coordinator.validator_model = "anthropic/claude-opus-4.7"
            proofs_route.coordinator.validator_provider = "openrouter"
            proofs_route.coordinator.validator_openrouter_provider = "Anthropic"
            proofs_route.coordinator.validator_openrouter_reasoning_effort = "xhigh"
            proofs_route.coordinator.validator_lm_studio_fallback = None
            proofs_route.coordinator.validator_context_window = 1000000
            proofs_route.coordinator.validator_max_tokens = 128000
            proofs_route.coordinator.validator_supercharge_enabled = False

            request = ProofCheckRequest(
                source_type="brainstorm",
                source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                proof_runtime_config=stale_request_snapshot,
            )
            snapshot = await proofs_route._get_runtime_snapshot(request)
        finally:
            for key, value in previous_state.items():
                setattr(proofs_route.coordinator, key, value)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.brainstorm.model_id, "anthropic/claude-opus-4.7")
        self.assertEqual(snapshot.validator.model_id, "anthropic/claude-opus-4.7")
        self.assertEqual(snapshot.validator.openrouter_provider, "Anthropic")

    async def test_manual_aggregator_runtime_does_not_fall_back_to_autonomous_snapshot(self):
        previous_submitters = proofs_route.coordinator.submitter_configs
        previous_validator_model = proofs_route.coordinator.validator_model
        try:
            proofs_route.coordinator.submitter_configs = []
            proofs_route.coordinator.validator_model = ""
            request = ProofCheckRequest(
                source_type="brainstorm",
                source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
            )
            with mock.patch.object(
                proofs_route.autonomous_coordinator,
                "get_proof_runtime_config",
                side_effect=AssertionError("manual sources must not read autonomous proof snapshots"),
            ):
                snapshot = await proofs_route._get_runtime_snapshot(request)
        finally:
            proofs_route.coordinator.submitter_configs = previous_submitters
            proofs_route.coordinator.validator_model = previous_validator_model

        self.assertIsNone(snapshot)

    async def test_manual_compiler_current_runtime_uses_rigor_and_proofs_submitter(self):
        previous_values = {
            "high_param_submitter": getattr(proofs_route.compiler_coordinator, "high_param_submitter", None),
            "validator_model": getattr(proofs_route.compiler_coordinator, "validator_model", None),
            "high_param_provider": getattr(proofs_route.compiler_coordinator, "high_param_provider", None),
            "high_param_openrouter_provider": getattr(proofs_route.compiler_coordinator, "high_param_openrouter_provider", None),
            "high_param_openrouter_reasoning_effort": getattr(proofs_route.compiler_coordinator, "high_param_openrouter_reasoning_effort", None),
            "high_param_lm_studio_fallback": getattr(proofs_route.compiler_coordinator, "high_param_lm_studio_fallback", None),
            "high_param_supercharge_enabled": getattr(proofs_route.compiler_coordinator, "high_param_supercharge_enabled", None),
            "validator_provider": getattr(proofs_route.compiler_coordinator, "validator_provider", None),
            "validator_openrouter_provider": getattr(proofs_route.compiler_coordinator, "validator_openrouter_provider", None),
            "validator_openrouter_reasoning_effort": getattr(proofs_route.compiler_coordinator, "validator_openrouter_reasoning_effort", None),
            "validator_lm_studio_fallback": getattr(proofs_route.compiler_coordinator, "validator_lm_studio_fallback", None),
            "validator_context_window": getattr(proofs_route.compiler_coordinator, "validator_context_window", None),
            "validator_max_tokens": getattr(proofs_route.compiler_coordinator, "validator_max_tokens", None),
            "validator_supercharge_enabled": getattr(proofs_route.compiler_coordinator, "validator_supercharge_enabled", None),
            "compiler_high_param_context_window": system_config.compiler_high_param_context_window,
            "compiler_high_param_max_output_tokens": system_config.compiler_high_param_max_output_tokens,
        }
        try:
            proofs_route.compiler_coordinator.high_param_submitter = SimpleNamespace(model_name="rigor-model")
            proofs_route.compiler_coordinator.validator_model = "validator-model"
            proofs_route.compiler_coordinator.high_param_provider = "openrouter"
            proofs_route.compiler_coordinator.high_param_openrouter_provider = "RigorHost"
            proofs_route.compiler_coordinator.high_param_openrouter_reasoning_effort = "xhigh"
            proofs_route.compiler_coordinator.high_param_lm_studio_fallback = "rigor-fallback"
            proofs_route.compiler_coordinator.high_param_supercharge_enabled = True
            proofs_route.compiler_coordinator.validator_provider = "openrouter"
            proofs_route.compiler_coordinator.validator_openrouter_provider = "ValidatorHost"
            proofs_route.compiler_coordinator.validator_openrouter_reasoning_effort = "high"
            proofs_route.compiler_coordinator.validator_lm_studio_fallback = "validator-fallback"
            proofs_route.compiler_coordinator.validator_context_window = 4096
            proofs_route.compiler_coordinator.validator_max_tokens = 512
            proofs_route.compiler_coordinator.validator_supercharge_enabled = False
            system_config.compiler_high_param_context_window = 8192
            system_config.compiler_high_param_max_output_tokens = 1024

            request = ProofCheckRequest(
                source_type="paper",
                source_id=proofs_route.MANUAL_COMPILER_CURRENT_SOURCE_ID,
            )
            snapshot = await proofs_route._get_runtime_snapshot(request)
        finally:
            for key, value in previous_values.items():
                if key.startswith("compiler_"):
                    setattr(system_config, key, value)
                else:
                    setattr(proofs_route.compiler_coordinator, key, value)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.paper.model_id, "rigor-model")
        self.assertEqual(snapshot.paper.provider, "openrouter")
        self.assertEqual(snapshot.paper.openrouter_provider, "RigorHost")
        self.assertEqual(snapshot.paper.openrouter_reasoning_effort, "xhigh")
        self.assertEqual(snapshot.paper.lm_studio_fallback_id, "rigor-fallback")
        self.assertEqual(snapshot.paper.context_window, 8192)
        self.assertEqual(snapshot.paper.max_output_tokens, 1024)
        self.assertTrue(snapshot.paper.supercharge_enabled)
        self.assertEqual(snapshot.validator.model_id, "validator-model")

    async def test_cross_source_exact_match_runs_novelty_and_stores_current_occurrence(self):
        existing_record = ProofRecord(
            proof_id="proof_existing",
            theorem_statement="Exact theorem sentinel.",
            source_type="brainstorm",
            source_id="topic_001",
            lean_code="theorem exact_theorem_sentinel : True := by trivial",
            novel=True,
            novelty_tier="mathematical_discovery",
        )

        class FakeProofDb:
            def __init__(self):
                self.records = [existing_record]

            async def get_all_proofs(self):
                return list(self.records)

            def get_novel_proofs_for_injection(self):
                return "Existing private-history proof context."

            async def add_proof_occurrence(self, record):
                stored = record.model_copy(update={"proof_id": "proof_current_occurrence"})
                self.records.append(stored)
                return stored

            async def get_or_create_active_run_id(self):
                return "manual-stable-run"

        novelty_calls = []

        async def assess_exact_match(**kwargs):
            novelty_calls.append(kwargs)
            return "not_novel", "Validator independently classified this occurrence."

        old_assess = proof_registration_module.assess_proof_novelty
        proof_registration_module.assess_proof_novelty = assess_exact_match
        try:
            registration = await proof_registration_module.register_verified_lean_proof(
                proof_database=FakeProofDb(),
                user_prompt="Prove the prompt.",
                theorem_statement=existing_record.theorem_statement,
                lean_code=existing_record.lean_code,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                task_id="proof_novelty_test",
                role_id="autonomous_proof_novelty",
                source_type="paper",
                source_id="paper_001",
            )
        finally:
            proof_registration_module.assess_proof_novelty = old_assess

        self.assertFalse(registration.duplicate)
        self.assertEqual(len(novelty_calls), 1)
        self.assertEqual(novelty_calls[0]["existing_novel_proofs"], "")
        self.assertEqual(registration.record.proof_id, "proof_current_occurrence")
        self.assertEqual(registration.record.novelty_tier, "not_novel")
        self.assertEqual(registration.record.independent_novelty_tier, "not_novel")
        self.assertEqual(
            registration.record.independent_novelty_reasoning,
            "Validator independently classified this occurrence.",
        )
        self.assertEqual(registration.record.exact_duplicate_proof_id, "proof_existing")
        self.assertEqual(registration.record.exact_duplicate_run_id, "brainstorm:topic_001")
        self.assertEqual(registration.record.run_id, "manual-stable-run")
        self.assertEqual(registration.record.user_prompt, "Prove the prompt.")

    async def test_cross_run_exact_novel_match_gets_silent_duplicate_overlay(self):
        existing_record = ProofRecord(
            proof_id="proof_existing",
            theorem_statement="Exact   theorem\n sentinel.",
            source_type="brainstorm",
            source_id="topic_001",
            run_id="prior-run",
            lean_code="\r\ntheorem exact_theorem_sentinel : True := by trivial\r\n",
            novel=True,
            novelty_tier="mathematical_discovery",
        )

        class FakeProofDb:
            def __init__(self):
                self.records = [existing_record]

            async def get_all_proofs(self):
                return list(self.records)

            def get_novel_proofs_for_injection(self):
                return ""

            async def add_proof_occurrence(self, record):
                stored = record.model_copy(update={"proof_id": "proof_current"})
                self.records.append(stored)
                return stored

        async def assess_exact_match(**_kwargs):
            return "novel_variant", "Independent novelty judgment."

        old_assess = proof_registration_module.assess_proof_novelty
        proof_registration_module.assess_proof_novelty = assess_exact_match
        try:
            registration = await proof_registration_module.register_verified_lean_proof(
                proof_database=FakeProofDb(),
                user_prompt="Prove the prompt.",
                theorem_statement=" Exact theorem sentinel. ",
                lean_code="theorem exact_theorem_sentinel : True := by trivial\n",
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                task_id="proof_novelty_test",
                role_id="autonomous_proof_novelty",
                source_type="paper",
                source_id="paper_001",
                run_id="current-run",
            )
        finally:
            proof_registration_module.assess_proof_novelty = old_assess

        self.assertFalse(registration.duplicate)
        self.assertEqual(registration.record.proof_id, "proof_current")
        self.assertEqual(registration.record.novelty_tier, "duplicate_novel")
        self.assertEqual(registration.record.independent_novelty_tier, "novel_variant")
        self.assertEqual(
            registration.record.independent_novelty_reasoning,
            "Independent novelty judgment.",
        )
        self.assertEqual(registration.record.exact_duplicate_proof_id, "proof_existing")
        self.assertEqual(registration.record.exact_duplicate_run_id, "prior-run")
        self.assertEqual(registration.record.artifact_purpose, "verified_occurrence")
        self.assertTrue(registration.record.canonical_identity_version)
        self.assertTrue(registration.record.canonical_theorem_statement_hash)
        self.assertTrue(registration.record.canonical_lean_code_hash)

    async def test_archived_cross_mode_exact_match_overlays_duplicate(self):
        from backend.shared.proof_search.models import UnifiedProofSearchRecord

        archived = UnifiedProofSearchRecord(
            search_id="manual:archive:proof_009",
            corpus="manual",
            corpus_scope="archived",
            source_kind="verified_proof",
            proof_id="proof_009",
            session_id="manual-archive",
            run_id="manual-prior-run",
            source_type="paper",
            source_id="manual-paper",
            theorem_statement="Archived exact theorem.",
            lean_code="theorem archived_exact : True := by trivial",
            novelty_tier="novel_variant",
            canonical_uri="moto-proof://manual/manual-archive/proof_009",
        )

        class FakeProofDb:
            async def get_all_proofs(self):
                return []

            async def add_proof_occurrence(self, record):
                return record.model_copy(update={"proof_id": "proof_current"})

        class FakeSearchService:
            async def exact_identity_neighborhood(self, **kwargs):
                self.kwargs = kwargs
                return [archived]

        async def assess_exact_match(**kwargs):
            self.assertEqual(kwargs["existing_novel_proofs"], "")
            return "mathematical_discovery", "Independent archived-match judgment."

        from backend.shared.proof_search import search_service as search_service_module

        old_assess = proof_registration_module.assess_proof_novelty
        old_service = search_service_module.proof_search_service
        proof_registration_module.assess_proof_novelty = assess_exact_match
        fake_service = FakeSearchService()
        search_service_module.proof_search_service = fake_service
        try:
            registration = await proof_registration_module.register_verified_lean_proof(
                proof_database=FakeProofDb(),
                user_prompt="Prove the prompt.",
                theorem_statement="Archived   exact theorem.",
                lean_code="\r\ntheorem archived_exact : True := by trivial\r\n",
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                task_id="proof_novelty_test",
                role_id="autonomous_proof_novelty",
                source_type="leanoj_final",
                source_id="leanoj-current",
                run_id="leanoj-current-run",
            )
        finally:
            proof_registration_module.assess_proof_novelty = old_assess
            search_service_module.proof_search_service = old_service

        self.assertEqual(registration.record.novelty_tier, "duplicate_novel")
        self.assertEqual(
            registration.record.independent_novelty_tier,
            "mathematical_discovery",
        )
        self.assertEqual(registration.record.exact_duplicate_proof_id, "proof_009")
        self.assertEqual(registration.record.exact_duplicate_run_id, "manual-prior-run")
        self.assertEqual(set(fake_service.kwargs["corpora"]), {"moto", "manual", "leanoj"})
        self.assertEqual(
            fake_service.kwargs["exclude_run_ids"],
            ["leanoj-current-run"],
        )

    async def test_duplicate_novel_registration_uses_known_style_broadcast_event(self):
        events = []
        record = ProofRecord(
            proof_id="proof_duplicate_novel",
            theorem_statement="Duplicate novel theorem sentinel.",
            source_type="paper",
            source_id="paper_001",
            lean_code="theorem duplicate_novel_event_sentinel : True := by trivial",
            novel=True,
            novelty_tier="duplicate_novel",
        )

        async def broadcast(event_type, payload):
            events.append((event_type, payload))

        await proof_registration_module._broadcast_registered_proof(
            broadcast_fn=broadcast,
            record=record,
            base_event={"source_type": "paper"},
        )

        self.assertEqual([event_type for event_type, _payload in events], ["known_proof_verified"])
        self.assertTrue(events[0][1]["is_novel"])
        self.assertEqual(events[0][1]["novelty_tier"], "duplicate_novel")

    async def test_proof_library_novel_category_uses_prompt_injection_tiers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProofDatabase()
            db.set_base_dir(Path(tmpdir))
            await db.initialize()

            records = [
                ProofRecord(
                    proof_id="",
                    theorem_statement="Legacy novel flag but not novel tier.",
                    source_type="brainstorm",
                    source_id="topic_legacy",
                    lean_code="theorem legacy_true_not_novel : True := by trivial",
                    novel=True,
                    novelty_tier="not_novel",
                ),
                ProofRecord(
                    proof_id="",
                    theorem_statement="Unknown legacy novel tier.",
                    source_type="brainstorm",
                    source_id="topic_unknown",
                    lean_code="theorem legacy_true_unknown : True := by trivial",
                    novel=True,
                    novelty_tier="novel",
                ),
                ProofRecord(
                    proof_id="",
                    theorem_statement="Strict novel theorem.",
                    source_type="brainstorm",
                    source_id="topic_novel",
                    lean_code="theorem strict_novel_variant : True := by trivial",
                    novel=True,
                    novelty_tier="novel_variant",
                ),
            ]
            for record in records:
                await db.add_proof(record)

            novel_entries = await db._list_proofs_from_directory(Path(tmpdir), "test_session", "novel")
            all_entries = await db._list_proofs_from_directory(Path(tmpdir), "test_session", "all")

        self.assertEqual(len(all_entries), 3)
        self.assertEqual([entry["theorem_statement"] for entry in novel_entries], ["Strict novel theorem."])

    async def test_cross_source_near_duplicate_still_runs_true_novelty_assessment(self):
        existing_record = ProofRecord(
            proof_id="proof_existing",
            theorem_statement="Near duplicate theorem sentinel.",
            source_type="brainstorm",
            source_id="topic_001",
            lean_code="theorem near_duplicate_sentinel : True := by trivial",
            novel=True,
            novelty_tier="mathematical_discovery",
        )

        class FakeProofDb:
            def __init__(self):
                self.records = [existing_record]

            async def get_all_proofs(self):
                return list(self.records)

            def get_novel_proofs_for_injection(self):
                return "existing novel context"

            async def add_proof_if_absent(self, record):
                stored = record.model_copy(update={"proof_id": "proof_near_duplicate"})
                self.records.append(stored)
                return stored, False

        novelty_calls = []

        async def assess_near_duplicate(**kwargs):
            novelty_calls.append(kwargs)
            return "novel_variant", "Near duplicate advances the user's problem-solving context."

        old_assess = proof_registration_module.assess_proof_novelty
        proof_registration_module.assess_proof_novelty = assess_near_duplicate
        try:
            registration = await proof_registration_module.register_verified_lean_proof(
                proof_database=FakeProofDb(),
                user_prompt="Prove the prompt.",
                theorem_statement="Near duplicate theorem sentinel with stronger context.",
                lean_code="theorem near_duplicate_sentinel_stronger : True := by trivial",
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                task_id="proof_novelty_test",
                role_id="autonomous_proof_novelty",
                source_type="paper",
                source_id="paper_001",
            )
        finally:
            proof_registration_module.assess_proof_novelty = old_assess

        self.assertEqual(len(novelty_calls), 1)
        self.assertTrue(registration.record.novel)
        self.assertEqual(registration.record.novelty_tier, "novel_variant")
        self.assertIn("advances", registration.record.novelty_reasoning)

    async def test_proof_novelty_json_parse_failure_retries_before_rating(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return {"choices": [{"message": {"content": '{"novelty_tier":'}}]}
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "novelty_tier": "mathematical_discovery",
                                        "reasoning": "Retry recovered a valid novelty judgement.",
                                    }
                                )
                            }
                        }
                    ]
                }

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_novelty_module.api_client_manager
        proof_novelty_module.api_client_manager = fake_api_client_manager
        try:
            novelty_tier, reasoning = await proof_novelty_module.assess_proof_novelty(
                user_prompt="Solve the prompt.",
                theorem_statement="A verified theorem.",
                lean_code="theorem retry_novelty_success : True := by trivial",
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                existing_novel_proofs="",
                task_id="proof_novelty_test",
            )
        finally:
            proof_novelty_module.api_client_manager = old_api_client_manager

        self.assertEqual(novelty_tier, "mathematical_discovery")
        self.assertIn("Retry recovered", reasoning)
        self.assertEqual(len(fake_api_client_manager.calls), 2)
        self.assertEqual(fake_api_client_manager.calls[1]["task_id"], "proof_novelty_test_retry")
        self.assertEqual(fake_api_client_manager.calls[1]["temperature"], 0.0)
        self.assertEqual(fake_api_client_manager.calls[1]["role_id"], "autonomous_proof_novelty")
        self.assertIn("previous proof-novelty response", fake_api_client_manager.calls[1]["messages"][-1]["content"])

    async def test_proof_novelty_invalid_tier_retries_before_downgrade(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "novelty_tier": "brand_new",
                                            "reasoning": "Bad tier should not be accepted.",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "novelty_tier": "novel_variant",
                                        "reasoning": "Retry selected a valid tier.",
                                    }
                                )
                            }
                        }
                    ]
                }

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_novelty_module.api_client_manager
        proof_novelty_module.api_client_manager = fake_api_client_manager
        try:
            novelty_tier, reasoning = await proof_novelty_module.assess_proof_novelty(
                user_prompt="Solve the prompt.",
                theorem_statement="A verified theorem.",
                lean_code="theorem retry_invalid_tier : True := by trivial",
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                existing_novel_proofs="",
                task_id="proof_novelty_bad_tier",
            )
        finally:
            proof_novelty_module.api_client_manager = old_api_client_manager

        self.assertEqual(novelty_tier, "novel_variant")
        self.assertIn("valid tier", reasoning)
        self.assertEqual(len(fake_api_client_manager.calls), 2)

    async def test_proof_novelty_model_cannot_assign_duplicate_overlay(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                tier = "duplicate_novel" if len(self.calls) == 1 else "novel_variant"
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "novelty_tier": tier,
                                        "reasoning": "Independent novelty only.",
                                    }
                                )
                            }
                        }
                    ]
                }

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_novelty_module.api_client_manager
        proof_novelty_module.api_client_manager = fake_api_client_manager
        try:
            novelty_tier, _ = await proof_novelty_module.assess_proof_novelty(
                user_prompt="Solve the prompt.",
                theorem_statement="A verified theorem.",
                lean_code="theorem reject_model_duplicate_overlay : True := by trivial",
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                existing_novel_proofs="",
                task_id="proof_novelty_duplicate_overlay",
            )
        finally:
            proof_novelty_module.api_client_manager = old_api_client_manager

        self.assertEqual(novelty_tier, "novel_variant")
        self.assertEqual(len(fake_api_client_manager.calls), 2)
        self.assertNotIn(
            '"not_novel | duplicate_novel',
            fake_api_client_manager.calls[1]["messages"][-1]["content"],
        )

    async def test_proof_novelty_missing_tier_retries_before_downgrade(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "reasoning": "Missing tier should not be accepted.",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "novelty_tier": "novel_formulation",
                                        "reasoning": "Retry supplied the missing tier.",
                                    }
                                )
                            }
                        }
                    ]
                }

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_novelty_module.api_client_manager
        proof_novelty_module.api_client_manager = fake_api_client_manager
        try:
            novelty_tier, reasoning = await proof_novelty_module.assess_proof_novelty(
                user_prompt="Solve the prompt.",
                theorem_statement="A verified theorem.",
                lean_code="theorem retry_missing_tier : True := by trivial",
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                existing_novel_proofs="",
                task_id="proof_novelty_missing_tier",
            )
        finally:
            proof_novelty_module.api_client_manager = old_api_client_manager

        self.assertEqual(novelty_tier, "novel_formulation")
        self.assertIn("missing tier", reasoning)
        self.assertEqual(len(fake_api_client_manager.calls), 2)

    async def test_proof_novelty_retry_exhaustion_downgrades_to_not_novel(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                return {"choices": [{"message": {"content": '{"novelty_tier":'}}]}

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_novelty_module.api_client_manager
        proof_novelty_module.api_client_manager = fake_api_client_manager
        try:
            novelty_tier, reasoning = await proof_novelty_module.assess_proof_novelty(
                user_prompt="Solve the prompt.",
                theorem_statement="A verified theorem.",
                lean_code="theorem retry_novelty_exhausted : True := by trivial",
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                existing_novel_proofs="",
                task_id="proof_novelty_exhausted",
            )
        finally:
            proof_novelty_module.api_client_manager = old_api_client_manager

        self.assertEqual(novelty_tier, "not_novel")
        self.assertIn("retry failed after retry exhaustion", reasoning)
        self.assertEqual(len(fake_api_client_manager.calls), 2)

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

    def test_formalization_prompt_includes_retrieved_proof_search_context(self):
        prompt = build_proof_formalization_prompt(
            user_prompt="Solve the actual user prompt.",
            source_type="paper",
            theorem_statement="Target theorem.",
            formal_sketch="Use the source.",
            full_source_content="Paper body without proof appendix.",
            source_excerpt="Local target excerpt.",
            prior_attempts=[],
            retrieved_proofs_context=(
                "Result 1\n"
                "Source: syntheticlib4 stable\n"
                "Theorem: retrieved_pattern\n"
                "Lean code hash: code_hash"
            ),
        )

        retrieved_section = prompt.split(
            "SYNTHETIC / LOCAL VERIFIED PROOF SEARCH RESULTS:",
            1,
        )[1].split("COMMON LEAN 4 PITFALLS TO AVOID:", 1)[0]

        self.assertIn("retrieved_pattern", retrieved_section)
        self.assertIn("Use retrieved proofs only as optional proof-pattern", retrieved_section)

    async def test_manual_aggregator_prompt_falls_back_to_persisted_prompt(self):
        old_data_dir = system_config.data_dir
        old_validator = proofs_route.coordinator.validator
        expected_prompt = "\nExact manual Aggregator prompt sentinel.\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                proofs_route.coordinator.validator = None
                await save_manual_aggregator_prompt(expected_prompt)

                recovered_prompt = await proofs_route._manual_aggregator_prompt()
            finally:
                await clear_manual_aggregator_prompt()
                proofs_route.coordinator.validator = old_validator
                system_config.data_dir = old_data_dir

        self.assertEqual(recovered_prompt, expected_prompt)

    async def test_manual_aggregator_prompt_rejects_empty_overwrite(self):
        old_data_dir = system_config.data_dir
        expected_prompt = "Durable manual Aggregator prompt sentinel."

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                await save_manual_aggregator_prompt(expected_prompt)
                await save_manual_aggregator_prompt("")

                recovered_prompt = await load_manual_aggregator_prompt()
            finally:
                await clear_manual_aggregator_prompt()
                system_config.data_dir = old_data_dir

        self.assertEqual(recovered_prompt, expected_prompt)

    async def test_manual_compiler_prompt_rejects_empty_overwrite(self):
        old_data_dir = system_config.data_dir
        expected_prompt = "Durable manual Compiler prompt sentinel."

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                await save_manual_compiler_prompt(expected_prompt)
                await save_manual_compiler_prompt("")

                recovered_prompt = await load_manual_compiler_prompt()
            finally:
                await clear_manual_compiler_prompt()
                system_config.data_dir = old_data_dir

        self.assertEqual(recovered_prompt, expected_prompt)

    async def test_manual_prompt_routes_return_persisted_prompts(self):
        old_data_dir = system_config.data_dir
        aggregator_prompt = "Route manual Aggregator prompt sentinel."
        compiler_prompt = "Route manual Compiler prompt sentinel."

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.data_dir = tmpdir
                await save_manual_aggregator_prompt(aggregator_prompt)
                await save_manual_compiler_prompt(compiler_prompt)

                aggregator_response = await aggregator_route.get_prompt()
                compiler_response = await compiler_route.get_prompt()
            finally:
                await clear_manual_aggregator_prompt()
                await clear_manual_compiler_prompt()
                system_config.data_dir = old_data_dir

        self.assertEqual(aggregator_response["prompt"], aggregator_prompt)
        self.assertEqual(compiler_response["prompt"], compiler_prompt)

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

    async def test_manual_aggregator_source_strips_appended_proofs(self):
        old_shared_training_file = system_config.shared_training_file
        old_memory_path = proofs_route.shared_training_memory.file_path
        old_insights = list(proofs_route.shared_training_memory.insights)
        old_proof_appendix = proofs_route.shared_training_memory.proof_appendix
        old_submission_count = proofs_route.shared_training_memory.submission_count
        old_last_ragged = proofs_route.shared_training_memory.last_ragged_submission_count
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                manual_path = Path(tmpdir) / "rag_shared_training.txt"
                manual_path.write_text(
                    "================================================================================\n"
                    "SUBMISSION #1 | Accepted: 2026-05-22T00:00:00\n"
                    "================================================================================\n\n"
                    "Manual aggregator body sentinel.\n\n"
                    "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
                    "Proof 1: Appended manual proof sentinel\n"
                    "Status: Verified (Novel)\n"
                    "Proof ID: proof_existing\n"
                    "Lean 4 Code:\n"
                    "theorem appended_manual_proof : True := by trivial\n"
                    "---\n",
                    encoding="utf-8",
                )
                system_config.shared_training_file = str(manual_path)
                proofs_route.shared_training_memory.file_path = manual_path
                await proofs_route.shared_training_memory.reload_insights_from_current_path()

                content, _title, _user_prompt = await proofs_route._resolve_manual_source(
                    SimpleNamespace(
                        source_type="brainstorm",
                        source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                    ),
                    _FakeActiveProofDatabase(),
                )
        finally:
            system_config.shared_training_file = old_shared_training_file
            proofs_route.shared_training_memory.file_path = old_memory_path
            proofs_route.shared_training_memory.insights = old_insights
            proofs_route.shared_training_memory.proof_appendix = old_proof_appendix
            proofs_route.shared_training_memory.submission_count = old_submission_count
            proofs_route.shared_training_memory.last_ragged_submission_count = old_last_ragged

        self.assertIn("Manual aggregator body sentinel.", content)
        self.assertNotIn("appended_manual_proof", content)

    async def test_manual_aggregator_source_uses_persisted_prompt_after_restart(self):
        old_data_dir = system_config.data_dir
        old_shared_training_file = system_config.shared_training_file
        old_memory_path = proofs_route.shared_training_memory.file_path
        old_insights = list(proofs_route.shared_training_memory.insights)
        old_proof_appendix = proofs_route.shared_training_memory.proof_appendix
        old_submission_count = proofs_route.shared_training_memory.submission_count
        old_last_ragged = proofs_route.shared_training_memory.last_ragged_submission_count
        old_validator = proofs_route.coordinator.validator
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                manual_path = Path(tmpdir) / "rag_shared_training.txt"
                manual_path.write_text(
                    "================================================================================\n"
                    "SUBMISSION #1 | Accepted: 2026-05-22T00:00:00\n"
                    "================================================================================\n\n"
                    "Manual aggregator body sentinel.\n",
                    encoding="utf-8",
                )
                system_config.data_dir = tmpdir
                system_config.shared_training_file = str(manual_path)
                proofs_route.shared_training_memory.file_path = manual_path
                proofs_route.coordinator.validator = None
                await save_manual_aggregator_prompt("Persisted manual prompt sentinel.")
                await proofs_route.shared_training_memory.reload_insights_from_current_path()

                _content, _title, user_prompt = await proofs_route._resolve_manual_source(
                    SimpleNamespace(
                        source_type="brainstorm",
                        source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                    ),
                    _FakeActiveProofDatabase(),
                )
                await clear_manual_aggregator_prompt()
        finally:
            system_config.data_dir = old_data_dir
            system_config.shared_training_file = old_shared_training_file
            proofs_route.shared_training_memory.file_path = old_memory_path
            proofs_route.shared_training_memory.insights = old_insights
            proofs_route.shared_training_memory.proof_appendix = old_proof_appendix
            proofs_route.shared_training_memory.submission_count = old_submission_count
            proofs_route.shared_training_memory.last_ragged_submission_count = old_last_ragged
            proofs_route.coordinator.validator = old_validator

        self.assertIn("ACTIVE_DB::Persisted manual prompt sentinel.", user_prompt)

    async def test_manual_aggregator_append_targets_manual_file_if_singleton_path_changed(self):
        old_shared_training_file = system_config.shared_training_file
        old_memory_path = proofs_route.shared_training_memory.file_path
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                manual_path = Path(tmpdir) / "rag_shared_training.txt"
                other_path = Path(tmpdir) / "brainstorm_topic_001.txt"
                manual_path.write_text(
                    "================================================================================\n"
                    "SUBMISSION #1 | Accepted: 2026-05-22T00:00:00\n"
                    "================================================================================\n\n"
                    "Manual aggregator body sentinel.\n",
                    encoding="utf-8",
                )
                other_path.write_text("Autonomous brainstorm sentinel.\n", encoding="utf-8")
                system_config.shared_training_file = str(manual_path)
                proofs_route.shared_training_memory.file_path = other_path

                record = ProofRecord(
                    proof_id="proof_manual_append",
                    theorem_statement="Manual append theorem sentinel.",
                    source_type="brainstorm",
                    source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                    lean_code="theorem manual_append_target : True := by trivial",
                    novel=True,
                    novelty_tier="mathematical_discovery",
                )
                await proofs_route._append_manual_aggregator_proof(record)
                manual_content = manual_path.read_text(encoding="utf-8")
                other_content = other_path.read_text(encoding="utf-8")
        finally:
            system_config.shared_training_file = old_shared_training_file
            proofs_route.shared_training_memory.file_path = old_memory_path

        self.assertIn("manual_append_target", manual_content)
        self.assertNotIn("manual_append_target", other_content)

    def test_duplicate_novel_proofs_can_repair_source_appendix_with_callback(self):
        async def append_callback(_proof):
            return None

        self.assertTrue(
            ProofVerificationStage._should_append_verified_proof(
                is_novel=True,
                duplicate=True,
                append_proof_callback=append_callback,
            )
        )
        self.assertFalse(
            ProofVerificationStage._should_append_verified_proof(
                is_novel=True,
                duplicate=True,
                append_proof_callback=None,
            )
        )

    def test_manual_checks_append_known_verified_proofs(self):
        self.assertTrue(
            ProofVerificationStage._should_append_verified_proof(
                is_novel=False,
                duplicate=False,
                append_proof_callback=None,
                append_known_proofs=ProofVerificationStage._should_append_known_proofs_for_trigger("manual"),
            )
        )
        self.assertFalse(
            ProofVerificationStage._should_append_verified_proof(
                is_novel=False,
                duplicate=False,
                append_proof_callback=None,
                append_known_proofs=ProofVerificationStage._should_append_known_proofs_for_trigger("automatic"),
            )
        )
        self.assertTrue(
            ProofVerificationStage._should_append_verified_proof(
                is_novel=False,
                duplicate=True,
                append_proof_callback=None,
                append_known_proofs=ProofVerificationStage._should_append_known_proofs_for_trigger("manual"),
            )
        )

    async def test_brainstorm_append_skips_existing_proof_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = BrainstormMemory()
            memory._base_dir = Path(tmpdir)
            db_path = Path(tmpdir) / "brainstorm_topic_001.txt"
            db_path.write_text("Brainstorm body sentinel.\n", encoding="utf-8")

            record = ProofRecord(
                proof_id="proof_known_append",
                theorem_statement="Known append theorem sentinel.",
                source_type="brainstorm",
                source_id="topic_001",
                lean_code="theorem known_append_target : True := by trivial",
                novel=False,
                novelty_tier="not_novel",
            )

            await memory.append_proofs_section("topic_001", record)
            await memory.append_proofs_section("topic_001", record)

            content = db_path.read_text(encoding="utf-8")

        self.assertEqual(content.count("Proof ID: proof_known_append"), 1)
        self.assertEqual(content.count("known_append_target"), 1)
        self.assertIn("Status: Verified (Known)", content)

    async def test_manual_compiler_current_append_updates_live_paper_once(self):
        old_paper_path = proofs_route.paper_memory.file_path
        old_version = proofs_route.paper_memory.version
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paper_path = Path(tmpdir) / "compiler_paper.txt"
                paper_path.write_text("Manual live paper sentinel.\n", encoding="utf-8")
                proofs_route.paper_memory.file_path = paper_path

                record = ProofRecord(
                    proof_id="proof_live_paper_known",
                    theorem_statement="Live paper known theorem sentinel.",
                    source_type="paper",
                    source_id=proofs_route.MANUAL_COMPILER_CURRENT_SOURCE_ID,
                    lean_code="theorem live_paper_known_target : True := by trivial",
                    novel=False,
                    novelty_tier="not_novel",
                )

                await proofs_route._append_manual_compiler_current_proof(record)
                await proofs_route._append_manual_compiler_current_proof(record)

                content = paper_path.read_text(encoding="utf-8")
        finally:
            proofs_route.paper_memory.file_path = old_paper_path
            proofs_route.paper_memory.version = old_version

        self.assertIn("Manual live paper sentinel.", content)
        self.assertIn("Live paper known theorem sentinel.", content)
        self.assertEqual(content.count("Theorem (proof_live_paper_known)"), 1)
        self.assertEqual(content.count("live_paper_known_target"), 1)

    async def test_compiler_proof_only_targets_manual_aggregator_source_for_append(self):
        captured = {}

        class FakeStage:
            async def run(self, **kwargs):
                captured.update(kwargs)

        async def fake_read_manual_aggregator_context():
            return "Manual Aggregator proof-only content sentinel."

        async def fake_append_callback(_proof):
            return True

        async def fake_broadcast(_event_type, _payload):
            return None

        old_stage = compiler_route.ProofVerificationStage
        old_read_context = compiler_route._read_manual_aggregator_context
        old_append = compiler_route.append_proof_to_manual_shared_training
        old_broadcast = compiler_route.websocket.broadcast_event
        try:
            compiler_route.ProofVerificationStage = FakeStage
            compiler_route._read_manual_aggregator_context = fake_read_manual_aggregator_context
            compiler_route.append_proof_to_manual_shared_training = fake_append_callback
            compiler_route.websocket.broadcast_event = fake_broadcast
            await compiler_route._run_compiler_aggregator_proof_check(
                SimpleNamespace(
                    compiler_prompt="Manual prompt",
                    writer_provider="lm_studio",
                    writer_model="writer-model",
                    writer_openrouter_provider=None,
                    writer_openrouter_reasoning_effort="auto",
                    writer_lm_studio_fallback=None,
                    writer_context_size=4000,
                    writer_max_output_tokens=1000,
                    writer_supercharge_enabled=False,
                    high_param_provider="lm_studio",
                    high_param_model="rigor-model",
                    high_param_openrouter_provider=None,
                    high_param_openrouter_reasoning_effort="auto",
                    high_param_lm_studio_fallback=None,
                    high_param_context_size=5000,
                    high_param_max_output_tokens=1200,
                    high_param_supercharge_enabled=False,
                    validator_provider="lm_studio",
                    validator_model="validator-model",
                    validator_openrouter_provider=None,
                    validator_openrouter_reasoning_effort="auto",
                    validator_lm_studio_fallback=None,
                    validator_context_size=4000,
                    validator_max_output_tokens=1000,
                    validator_supercharge_enabled=False,
                    assistant_provider="lm_studio",
                    assistant_model=None,
                    assistant_openrouter_provider=None,
                    assistant_openrouter_reasoning_effort="auto",
                    assistant_lm_studio_fallback=None,
                    assistant_context_size=4000,
                    assistant_max_output_tokens=1000,
                    assistant_supercharge_enabled=False,
                )
            )
        finally:
            compiler_route.ProofVerificationStage = old_stage
            compiler_route._read_manual_aggregator_context = old_read_context
            compiler_route.append_proof_to_manual_shared_training = old_append
            compiler_route.websocket.broadcast_event = old_broadcast

        self.assertEqual(captured["source_type"], "brainstorm")
        self.assertEqual(captured["source_id"], proofs_route.MANUAL_AGGREGATOR_SOURCE_ID)
        self.assertEqual(captured["submitter_model"], "rigor-model")
        self.assertEqual(captured["submitter_context"], 5000)
        self.assertEqual(captured["submitter_max_tokens"], 1200)
        self.assertFalse(captured["append_to_source"])
        self.assertIs(captured["append_proof_callback"], fake_append_callback)
        self.assertIn("Manual Aggregator proof-only content sentinel.", captured["content"])

    async def test_compiler_manual_aggregator_context_strips_appended_proofs(self):
        old_shared_training_file = system_config.shared_training_file
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                manual_path = Path(tmpdir) / "rag_shared_training.txt"
                manual_path.write_text(
                    "Manual Aggregator proof-only body sentinel.\n\n"
                    "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
                    "Lean 4 Code:\n"
                    "theorem compiler_context_duplicate : True := by trivial\n",
                    encoding="utf-8",
                )
                system_config.shared_training_file = str(manual_path)

                content = await compiler_route._read_manual_aggregator_context()
        finally:
            system_config.shared_training_file = old_shared_training_file

        self.assertIn("Manual Aggregator proof-only body sentinel.", content)
        self.assertNotIn("compiler_context_duplicate", content)

    async def test_saved_compiler_paper_append_helper_writes_novel_proof_once(self):
        record = ProofRecord(
            proof_id="proof_saved_append",
            theorem_statement="Saved compiler append theorem sentinel.",
            source_type="paper",
            source_id="compiler_manual_hash",
            lean_code="theorem saved_append_theorem : True := by trivial",
            novel=True,
            novelty_tier="mathematical_discovery",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = Path(tmpdir) / "compiler_paper_saved.txt"
            saved_path.write_text("Saved manual compiler paper body.\n", encoding="utf-8")

            self.assertTrue(
                await compiler_route._append_proof_to_saved_compiler_paper(saved_path, record)
            )
            self.assertTrue(
                await compiler_route._append_proof_to_saved_compiler_paper(saved_path, record)
            )
            content = saved_path.read_text(encoding="utf-8")

        self.assertIn("Saved compiler append theorem sentinel.", content)
        self.assertIn("saved_append_theorem", content)
        self.assertEqual(content.count("theorem saved_append_theorem"), 1)

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

    async def test_saved_compiler_proof_content_includes_full_stripped_aggregator_context(self):
        long_middle = "A" * 60000
        old_read_manual_aggregator_context = compiler_route._read_manual_aggregator_context

        async def full_aggregator_context():
            return (
                "Aggregator head sentinel.\n"
                f"{long_middle}\n"
                "Aggregator tail sentinel beyond old cap.\n\n"
                "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n"
                "theorem saved_compiler_context_duplicate : True := by trivial\n"
            ).split("=== PROOFS GENERATED FROM THIS BRAINSTORM", 1)[0].rstrip()

        compiler_route._read_manual_aggregator_context = full_aggregator_context
        try:
            content = await compiler_route._build_saved_compiler_proof_content(
                "Manual compiler paper body sentinel."
            )
        finally:
            compiler_route._read_manual_aggregator_context = old_read_manual_aggregator_context

        self.assertIn("Manual compiler paper body sentinel.", content)
        self.assertIn("Aggregator head sentinel.", content)
        self.assertIn("Aggregator tail sentinel beyond old cap.", content)
        self.assertNotIn("source context truncated", content)
        self.assertNotIn("saved_compiler_context_duplicate", content)

    async def test_proof_identification_requires_relevance_novelty_and_standard_rationales(self):
        class FakeApiClientManager:
            async def prewarm_assistant_memory_context(self, **_kwargs):
                return ""

            async def generate_completion(self, **_kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "has_provable_theorems": True,
                                        "theorems": [
                                            {
                                                "theorem_id": "missing_rationales",
                                                "statement": "A theorem with missing rationale.",
                                                "expected_novelty_tier": "mathematical_discovery",
                                                "prompt_relevance_rationale": "",
                                                "novelty_rationale": "Novel.",
                                                "why_not_standard_known_result": "Not standard.",
                                            },
                                            {
                                                "theorem_id": "valid",
                                                "statement": "A theorem that directly solves the prompt.",
                                                "expected_novelty_tier": "mathematical_discovery",
                                                "prompt_relevance_rationale": "This directly solves the user prompt.",
                                                "novelty_rationale": "Novel.",
                                                "why_not_standard_known_result": "Not standard.",
                                            },
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }

        old_api_client_manager = proof_identification_module.api_client_manager
        proof_identification_module.api_client_manager = FakeApiClientManager()
        try:
            agent = proof_identification_module.ProofIdentificationAgent(
                model_id="model",
                context_window=8000,
                max_output_tokens=1000,
                role_id="autonomous_proof_identification_test",
            )
            has_candidates, candidates = await agent.identify_candidates(
                user_research_prompt="Solve the prompt.",
                source_type="brainstorm",
                source_id="manual_aggregator",
                source_content="Source content.",
            )
        finally:
            proof_identification_module.api_client_manager = old_api_client_manager

        self.assertTrue(has_candidates)
        self.assertEqual([candidate.theorem_id for candidate in candidates], ["valid"])

    async def test_proof_identification_retries_codex_max_output_truncation(self):
        class FakeApiClientManager:
            def __init__(self):
                self.calls = []

            async def prewarm_assistant_memory_context(self, **_kwargs):
                return ""

            async def generate_completion(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise RuntimeError(
                        "OpenAI Codex completion failed: "
                        '{"type":"response.incomplete","response":{"status":"incomplete",'
                        '"incomplete_details":{"reason":"max_output_tokens"}}}'
                    )
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "has_provable_theorems": True,
                                        "theorems": [
                                            {
                                                "theorem_id": "retry_valid",
                                                "statement": "A retry-discovered theorem.",
                                                "formal_sketch": "Sketch.",
                                                "expected_novelty_tier": "novel_variant",
                                                "prompt_relevance_rationale": "It directly advances the prompt.",
                                                "novelty_rationale": "It is not routine.",
                                                "why_not_standard_known_result": "It is not a standard reference result.",
                                            }
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }

        fake_api_client_manager = FakeApiClientManager()
        old_api_client_manager = proof_identification_module.api_client_manager
        proof_identification_module.api_client_manager = fake_api_client_manager
        try:
            agent = proof_identification_module.ProofIdentificationAgent(
                model_id="gpt-5.5",
                context_window=8000,
                max_output_tokens=1000,
                role_id="autonomous_proof_identification_test",
            )
            has_candidates, candidates = await agent.identify_candidates(
                user_research_prompt="Solve the prompt.",
                source_type="brainstorm",
                source_id="manual_aggregator",
                source_content="Source content.",
            )
        finally:
            proof_identification_module.api_client_manager = old_api_client_manager

        self.assertTrue(has_candidates)
        self.assertEqual([candidate.theorem_id for candidate in candidates], ["retry_valid"])
        self.assertEqual(len(fake_api_client_manager.calls), 2)
        self.assertEqual(fake_api_client_manager.calls[1]["task_id"], "proof_id_000_retry")

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

        self.assertFalse(result.had_error)
        self.assertEqual(result.deferred_candidate_ids, ["overflow"])
        self.assertEqual(record_failed_calls, 0)
        self.assertIn("proof_context_overflow", events)
        self.assertNotIn("proof_attempts_exhausted", events)

    async def test_codex_max_output_incomplete_counts_as_failed_attempt(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        attempt_failure_summaries: list[str] = []
        checkpoint_statuses: list[str] = []
        candidate = ProofCandidate(
            theorem_id="codex_incomplete",
            statement="Codex incomplete theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        async def broadcast(event_type, payload):
            events.append(event_type)
            if event_type == "proof_attempt_failed":
                attempt_failure_summaries.append(payload.get("error_summary", ""))

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

        self.assertFalse(result.had_error)
        self.assertEqual(result.verified_count, 0)
        self.assertEqual(result.total_candidates, 1)
        self.assertIn("running", checkpoint_statuses)
        self.assertNotIn("error", checkpoint_statuses)
        self.assertEqual(events.count("proof_attempt_failed"), 5)
        self.assertTrue(any("MODEL OUTPUT TRUNCATED" in summary for summary in attempt_failure_summaries))
        self.assertIn("proof_check_complete", events)
        self.assertIn("proof_attempts_exhausted", events)

    async def test_chat_finish_reason_length_counts_as_failed_attempt(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        attempt_failure_summaries: list[str] = []
        candidate = ProofCandidate(
            theorem_id="chat_length_stop",
            statement="Chat length stop theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        async def broadcast(event_type, payload):
            events.append(event_type)
            if event_type == "proof_attempt_failed":
                attempt_failure_summaries.append(payload.get("error_summary", ""))

        async def fake_prepare_candidate(**kwargs):
            return kwargs["theorem_candidate"]

        async def fake_smt_check(**_kwargs):
            return None

        async def return_length_stopped_response(*_args, **_kwargs):
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{\"lean_code\": \""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1000, "total_tokens": 1010},
            }

        class FakeProofDb:
            pass

        old_generate_completion = proof_formalization_module.api_client_manager.generate_completion
        proof_formalization_module.api_client_manager.generate_completion = return_length_stopped_response
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
            )
        finally:
            proof_formalization_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertFalse(result.had_error)
        self.assertEqual(events.count("proof_attempt_failed"), 5)
        self.assertTrue(any("MODEL OUTPUT TRUNCATED" in summary for summary in attempt_failure_summaries))
        self.assertIn("proof_attempts_exhausted", events)
        self.assertIn("proof_check_complete", events)

    async def test_codex_transient_gateway_disconnect_preserves_proof_checkpoint(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []
        candidate = ProofCandidate(
            theorem_id="codex_gateway_timeout",
            statement="Codex transient theorem.",
            expected_novelty_tier="mathematical_discovery",
        )

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def fake_prepare_candidate(**kwargs):
            return kwargs["theorem_candidate"]

        async def fake_smt_check(**_kwargs):
            return None

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def raise_codex_gateway_timeout(*_args, **_kwargs):
            raise RuntimeError(
                "OpenAI Codex failed for role 'autonomous_proof_formalization_brainstorm' "
                "and no LM Studio fallback is configured: OpenAI Codex completion failed: "
                "upstream connect error or disconnect/reset before headers. "
                "retried and the latest reset reason: connection timeout"
            )

        class FakeProofDb:
            async def record_failed_candidate(self, *_args, **_kwargs):
                pass

        old_generate_completion = proof_formalization_module.api_client_manager.generate_completion
        proof_formalization_module.api_client_manager.generate_completion = raise_codex_gateway_timeout
        stage._prepare_candidate = fake_prepare_candidate
        stage._run_smt_check = fake_smt_check
        try:
            with self.assertRaises(RetryableProviderError) as exc:
                await stage.run(
                    content="brainstorm source",
                    source_type="brainstorm",
                    source_id="topic_001",
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

        self.assertIn("TRANSIENT PROVIDER ERROR", str(exc.exception))
        self.assertIn("provider_paused", checkpoint_statuses)
        self.assertNotIn("proof_attempts_exhausted", events)
        self.assertNotIn("proof_check_complete", events)

    async def test_codex_transient_identification_error_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def raise_codex_gateway_timeout(*_args, **_kwargs):
            raise RuntimeError(
                "OpenAI Codex failed for role 'autonomous_proof_identification_brainstorm' "
                "and no LM Studio fallback is configured: OpenAI Codex completion failed: "
                "upstream connect error or disconnect/reset before headers. "
                "retried and the latest reset reason: connection timeout"
            )

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = raise_codex_gateway_timeout
        try:
            with self.assertRaises(RetryableProviderError) as exc:
                await stage.run(
                    content="brainstorm source",
                    source_type="brainstorm",
                    source_id="topic_identification_timeout",
                    user_prompt="prove things",
                    submitter_model="model",
                    submitter_context=4000,
                    submitter_max_tokens=1000,
                    validator_model="validator",
                    validator_context=4000,
                    validator_max_tokens=1000,
                    broadcast_fn=broadcast,
                    novel_proofs_db=FakeProofDb(),
                    append_to_source=False,
                    checkpoint_callback=checkpoint,
                )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertIn("OpenAI Codex failed", str(exc.exception))
        self.assertEqual(checkpoint_statuses, [])
        self.assertNotIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_malformed_identification_output_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def malformed_identification_output(*_args, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "The brainstorm contains proof-relevant ideas, but this is not JSON."
                        }
                    }
                ]
            }

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = malformed_identification_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_malformed",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("No JSON found", result.error_message)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_empty_array_identification_output_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def empty_array_identification_output(*_args, **_kwargs):
            return {"choices": [{"message": {"content": "[]"}}]}

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = empty_array_identification_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_empty_array",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_schema_invalid_identification_output_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def schema_invalid_identification_output(*_args, **_kwargs):
            return {"choices": [{"message": {"content": json.dumps({"theorems": []})}}]}

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = schema_invalid_identification_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_schema_invalid",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("omitted has_provable_theorems", result.error_message)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_inconsistent_identification_output_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def inconsistent_identification_output(*_args, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"has_provable_theorems": True, "theorems": []}
                            )
                        }
                    }
                ]
            }

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = inconsistent_identification_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_inconsistent",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("claimed provable theorems", result.error_message)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_malformed_claimed_candidate_does_not_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def malformed_claimed_candidate_output(*_args, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "has_provable_theorems": True,
                                    "theorems": [
                                        {
                                            "theorem_id": "missing_rationales",
                                            "statement": "A claimed theorem missing rationales.",
                                            "expected_novelty_tier": "mathematical_discovery",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = malformed_claimed_candidate_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_malformed_candidate",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertTrue(result.had_error)
        self.assertIn("no valid theorem candidates", result.error_message)
        self.assertIn("error", checkpoint_statuses)
        self.assertIn("proof_check_complete", events)
        self.assertNotIn("proof_check_no_candidates", events)

    async def test_not_novel_identification_candidate_can_mark_no_candidates(self):
        old_lean4_enabled = system_config.lean4_enabled
        system_config.lean4_enabled = True
        stage = ProofVerificationStage()
        events: list[str] = []
        checkpoint_statuses: list[str] = []

        async def broadcast(event_type, _payload):
            events.append(event_type)

        async def checkpoint(payload):
            checkpoint_statuses.append(payload["status"])

        async def not_novel_identification_output(*_args, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "has_provable_theorems": True,
                                    "theorems": [
                                        {
                                            "theorem_id": "known",
                                            "statement": "A standard known theorem.",
                                            "expected_novelty_tier": "not_novel",
                                            "prompt_relevance_rationale": "It is adjacent to the prompt.",
                                            "novelty_rationale": "It is not novel.",
                                            "why_not_standard_known_result": "It is standard.",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }

        class FakeProofDb:
            pass

        old_generate_completion = proof_identification_module.api_client_manager.generate_completion
        proof_identification_module.api_client_manager.generate_completion = not_novel_identification_output
        try:
            result = await stage.run(
                content="brainstorm source",
                source_type="brainstorm",
                source_id="topic_identification_not_novel",
                user_prompt="prove things",
                submitter_model="model",
                submitter_context=4000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=4000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast,
                novel_proofs_db=FakeProofDb(),
                append_to_source=False,
                checkpoint_callback=checkpoint,
            )
        finally:
            proof_identification_module.api_client_manager.generate_completion = old_generate_completion
            system_config.lean4_enabled = old_lean4_enabled

        self.assertFalse(result.had_error)
        self.assertEqual(result.total_candidates, 0)
        self.assertIn("no_candidates", checkpoint_statuses)
        self.assertIn("proof_check_no_candidates", events)


class AutonomousProofFailedHintCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_all_data_removes_failed_hints_but_keeps_verified_proof_history(self):
        coordinator_module = importlib.import_module("backend.autonomous.core.autonomous_coordinator")
        queue_manager_module = importlib.import_module("backend.aggregator.core.queue_manager")
        from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            sessions_dir = data_root / "auto_sessions"
            session_dir = sessions_dir / "cleanup_session"
            proofs_dir = session_dir / "proofs"
            failed_dir = proofs_dir / "failed"
            failed_dir.mkdir(parents=True)
            (session_dir / "session_metadata.json").write_text(
                json.dumps({"session_id": "cleanup_session", "user_prompt": "Prompt"}),
                encoding="utf-8",
            )
            (session_dir / "session_stats.json").write_text(
                json.dumps({"current_brainstorm_id": "topic_1", "current_paper_id": "paper_1"}),
                encoding="utf-8",
            )
            (session_dir / "workflow_state.json").write_text(
                json.dumps({"current_tier": "tier1_aggregation"}),
                encoding="utf-8",
            )
            (proofs_dir / "proofs_index.json").write_text(
                json.dumps({"next_proof_id": 2, "proofs": [{"proof_id": "proof_001"}]}),
                encoding="utf-8",
            )
            (proofs_dir / "proof_001.json").write_text(
                json.dumps({"proof_id": "proof_001", "theorem_statement": "Verified"}),
                encoding="utf-8",
            )
            (proofs_dir / "proof_001_lean.lean").write_text(
                "theorem verified_cleanup : True := by trivial",
                encoding="utf-8",
            )
            (failed_dir / "topic_1.json").write_text(
                json.dumps({"items": [{"theorem_id": "failed_1"}]}),
                encoding="utf-8",
            )

            old_values = {
                "auto_sessions_base_dir": system_config.auto_sessions_base_dir,
                "auto_brainstorms_dir": system_config.auto_brainstorms_dir,
                "auto_papers_dir": system_config.auto_papers_dir,
                "auto_research_topic_rejections_file": system_config.auto_research_topic_rejections_file,
                "data_dir": system_config.data_dir,
            }
            try:
                system_config.data_dir = str(data_root)
                system_config.auto_sessions_base_dir = str(sessions_dir)
                system_config.auto_brainstorms_dir = str(data_root / "auto_brainstorms")
                system_config.auto_papers_dir = str(data_root / "auto_papers")
                system_config.auto_research_topic_rejections_file = str(
                    data_root / "auto_research_topic_rejections.txt"
                )

                coordinator = AutonomousCoordinator()
                with (
                    mock.patch.object(coordinator_module.session_manager, "clear", new=mock.AsyncMock()),
                    mock.patch.object(coordinator_module.brainstorm_memory, "set_session_manager"),
                    mock.patch.object(coordinator_module.paper_library, "set_session_manager"),
                    mock.patch.object(coordinator_module.research_metadata, "set_session_manager"),
                    mock.patch.object(coordinator_module.final_answer_memory, "set_session_manager"),
                    mock.patch.object(coordinator_module.proof_database, "set_session_manager"),
                    mock.patch.object(
                        coordinator_module.proof_database,
                        "clear_failed_candidates",
                        new=mock.AsyncMock(),
                    ) as clear_active_failed,
                    mock.patch.object(
                        coordinator_module.research_metadata,
                        "clear_all",
                        new=mock.AsyncMock(),
                    ),
                    mock.patch.object(
                        coordinator_module.autonomous_rejection_logs,
                        "clear_all",
                        new=mock.AsyncMock(),
                    ),
                    mock.patch.object(
                        coordinator_module.autonomous_api_logger,
                        "clear_logs",
                        new=mock.AsyncMock(),
                    ),
                    mock.patch.object(coordinator_module.autonomous_rag_manager, "reset"),
                    mock.patch.object(
                        coordinator_module.rag_manager,
                        "clear_all_documents_async",
                        new=mock.AsyncMock(),
                    ),
                    mock.patch.object(queue_manager_module.queue_manager, "clear", new=mock.AsyncMock()),
                ):
                    await coordinator.clear_all_data()

                metadata = json.loads((session_dir / "session_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "cleared")
                self.assertTrue(metadata["resume_disabled"])
                self.assertFalse((session_dir / "workflow_state.json").exists())
                self.assertFalse(failed_dir.exists())
                self.assertTrue((proofs_dir / "proofs_index.json").exists())
                self.assertTrue((proofs_dir / "proof_001.json").exists())
                self.assertTrue((proofs_dir / "proof_001_lean.lean").exists())
                clear_active_failed.assert_awaited_once()
            finally:
                for name, value in old_values.items():
                    setattr(system_config, name, value)


if __name__ == "__main__":
    unittest.main()
