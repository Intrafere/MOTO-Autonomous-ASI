import json
import unittest
from types import SimpleNamespace

from backend.autonomous.core import proof_registration as proof_registration_module
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.validation import compiler_validator as validator_module
from backend.compiler.validation.compiler_validator import CompilerValidator
from backend.compiler.memory.paper_memory import (
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
)
from backend.shared.config import system_config
from backend.shared.models import CompilerSubmission, ProofRecord
from backend.shared.solution_path import integration as solution_path_integration


def _set_compiler_test_token_budgets():
    old_values = (
        system_config.compiler_validator_context_window,
        system_config.compiler_validator_max_output_tokens,
        system_config.compiler_high_param_context_window,
        system_config.compiler_high_param_max_output_tokens,
    )
    system_config.compiler_validator_context_window = 20000
    system_config.compiler_validator_max_output_tokens = 1000
    system_config.compiler_high_param_context_window = 20000
    system_config.compiler_high_param_max_output_tokens = 1000
    return old_values


def _restore_compiler_test_token_budgets(old_values) -> None:
    (
        system_config.compiler_validator_context_window,
        system_config.compiler_validator_max_output_tokens,
        system_config.compiler_high_param_context_window,
        system_config.compiler_high_param_max_output_tokens,
    ) = old_values


class LeanPlacementValidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_lean_placement_prompt_and_forced_rigor_check(self) -> None:
        old_token_budgets = _set_compiler_test_token_budgets()
        validator = CompilerValidator(model_name="test-model", user_prompt="Write a paper.")
        submission = CompilerSubmission(
            submission_id="sub-lean-placement",
            mode="rigor",
            content="Anchor paragraph.\n\nTheorem text. Verified with Lean 4.",
            operation="insert_after",
            old_string="Anchor paragraph.",
            new_string="Theorem text. Verified with Lean 4.",
            reasoning="Place the theorem after the anchor.",
            metadata={
                "rigor_mode": "lean_placement",
                "theorem_statement": "theorem t : True",
                "lean_code": "theorem t : True := by trivial",
                "placement_attempt": 1,
            },
        )

        captured_prompt = {}
        hook_calls = []
        enqueue_calls = []

        async def fake_ensure_markers_intact() -> bool:
            return False

        async def fake_generate_completion(**kwargs):
            captured_prompt["text"] = kwargs["messages"][0]["content"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "reject",
                                    "reasoning": "Placement needs a better narrative bridge.",
                                    "coherence_check": True,
                                    "rigor_check": False,
                                    "placement_check": False,
                                    "solution_path_update": {
                                        "main_route": "Must be ignored for placement validation.",
                                        "route": {"goal": "Must not enqueue."},
                                    },
                                }
                            )
                        }
                    }
                ]
            }

        def fake_with_validator_hook(prompt, manager):
            hook_calls.append((prompt, manager))
            return prompt + "\nsolution_path_update advisory proposal instruction"

        async def fake_enqueue_optional_update(*args, **kwargs):
            enqueue_calls.append((args, kwargs))

        original_ensure = validator_module.paper_memory.ensure_markers_intact
        original_generate = validator_module.api_client_manager.generate_completion
        original_hook = solution_path_integration.with_validator_hook
        original_enqueue = solution_path_integration.enqueue_optional_update
        try:
            validator_module.paper_memory.ensure_markers_intact = fake_ensure_markers_intact
            validator_module.api_client_manager.generate_completion = fake_generate_completion
            solution_path_integration.with_validator_hook = fake_with_validator_hook
            solution_path_integration.enqueue_optional_update = fake_enqueue_optional_update

            result = await validator.validate_submission(
                submission,
                current_paper="Anchor paragraph.\n",
                current_outline="I. Outline",
            )
        finally:
            validator_module.paper_memory.ensure_markers_intact = original_ensure
            validator_module.api_client_manager.generate_completion = original_generate
            solution_path_integration.with_validator_hook = original_hook
            solution_path_integration.enqueue_optional_update = original_enqueue
            _restore_compiler_test_token_budgets(old_token_budgets)

        self.assertEqual(result.decision, "reject")
        self.assertTrue(result.rigor_check)
        self.assertFalse(result.placement_check)
        self.assertIn("Lean 4 Verified Theorem Placement", captured_prompt["text"])
        self.assertIn("MUST NOT re-evaluate", captured_prompt["text"])
        self.assertIn("LEAN 4 VERIFICATION CERTIFICATE", captured_prompt["text"])
        self.assertNotIn("solution_path_update", captured_prompt["text"])
        self.assertNotIn("advisory proposal instruction", captured_prompt["text"])
        self.assertEqual(hook_calls, [])
        self.assertEqual(enqueue_calls, [])

    def test_ordinary_validator_prompt_exposes_theorem_appendix_boundaries(self) -> None:
        validator = CompilerValidator(model_name="test-model", user_prompt="Write a paper.")
        prompt = validator._get_paper_validation_system_prompt("construction")

        self.assertIn(THEOREMS_APPENDIX_START, prompt)
        self.assertIn(THEOREMS_APPENDIX_END, prompt)
        self.assertIn("crosses either theorem-appendix boundary", prompt)
        self.assertIn("ordinary paper content inside the theorem appendix", prompt)

    def test_outline_prompt_does_not_require_optional_abstract(self) -> None:
        validator = CompilerValidator(model_name="test-model", user_prompt="Write a paper.")
        prompt = validator._get_outline_validation_system_prompt("outline_create")

        self.assertIn("Abstract is optional", prompt)
        self.assertNotIn("MISSING_REQUIRED_SECTION - Abstract", prompt)
        self.assertIn("MISSING_REQUIRED_SECTION - Introduction", prompt)


class HighParamRigorProofRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rigor_registration_uses_configured_paper_source(self) -> None:
        old_token_budgets = _set_compiler_test_token_budgets()
        submitter = HighParamSubmitter(
            model_name="rigor-model",
            user_prompt="Write a proof-rich paper.",
            validator_model="validator-model",
            validator_context_window=4000,
            validator_max_tokens=500,
        )
        submitter.set_rigor_proof_source("paper_123", "A Paper With Rigor Proofs")
        captured = {}

        async def fake_register_verified_lean_proof(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                record=ProofRecord(
                    proof_id="proof_123",
                    theorem_statement=kwargs["theorem_statement"],
                    source_type=kwargs["source_type"],
                    source_id=kwargs["source_id"],
                    source_title=kwargs["source_title"],
                    lean_code=kwargs["lean_code"],
                    novel=True,
                    novelty_tier="novel_variant",
                    novelty_reasoning="Ranked by the proof ranker.",
                ),
                duplicate=False,
            )

        original_register = proof_registration_module.register_verified_lean_proof
        try:
            proof_registration_module.register_verified_lean_proof = fake_register_verified_lean_proof
            is_novel, reasoning, stored, duplicate = await submitter._step_assess_novelty_and_store(
                theorem_statement="Every test theorem is indexed.",
                theorem_name="test_theorem",
                lean_code="theorem test_theorem : True := by trivial",
                formal_sketch="Trivial.",
                attempts=[],
                verification_notes="Test proof.",
            )
        finally:
            proof_registration_module.register_verified_lean_proof = original_register
            _restore_compiler_test_token_budgets(old_token_budgets)

        self.assertTrue(is_novel)
        self.assertFalse(duplicate)
        self.assertEqual(reasoning, "Ranked by the proof ranker.")
        self.assertEqual(stored.source_type, "paper")
        self.assertEqual(stored.source_id, "paper_123")
        self.assertEqual(stored.source_title, "A Paper With Rigor Proofs")
        self.assertEqual(captured["role_id"], "compiler_rigor_novelty")

    async def test_rigor_success_broadcasts_counted_proof_completion(self) -> None:
        old_token_budgets = _set_compiler_test_token_budgets()
        submitter = HighParamSubmitter(
            model_name="rigor-model",
            user_prompt="Write a proof-rich paper.",
            validator_model="validator-model",
            validator_context_window=4000,
            validator_max_tokens=500,
        )
        submitter.set_rigor_proof_source("paper_456", "Counted Rigor Paper")
        events = []

        async def capture_event(event, data):
            events.append((event, data))

        async def fake_discovery():
            return {
                "theorem_statement": "True is true.",
                "formal_sketch": "By trivial.",
                "source_excerpt": "Paper excerpt.",
                "theorem_origin": "existing_paper_claim",
                "placement_preference": "appendix_only",
                "expected_novelty_tier": "novel_variant",
                "prompt_relevance_rationale": "It checks the paper proof path.",
                "novelty_rationale": "It is a regression target.",
                "why_not_standard_known_result": "It is not intended as mathematics.",
            }

        async def fake_formalize(_candidate, _theorem_statement):
            integrity = SimpleNamespace(
                actual_theorem_statement="",
                actual_theorem_name="",
                category="aligned",
                reason="",
                downshift_reason="",
            )
            return "test_theorem", "theorem test_theorem : True := by trivial", [], integrity

        async def fake_assess(**_kwargs):
            return (
                True,
                "Ranked as a regression proof.",
                ProofRecord(
                    proof_id="proof_456",
                    theorem_statement="True is true.",
                    theorem_name="test_theorem",
                    source_type="paper",
                    source_id="paper_456",
                    source_title="Counted Rigor Paper",
                    lean_code="theorem test_theorem : True := by trivial",
                    novel=True,
                    novelty_tier="novel_variant",
                    novelty_reasoning="Ranked as a regression proof.",
                ),
                False,
            )

        original_lean_enabled = system_config.lean4_enabled
        try:
            system_config.lean4_enabled = True
            submitter.websocket_broadcaster = capture_event
            submitter._step_discovery = fake_discovery
            submitter._step_formalize = fake_formalize
            submitter._step_assess_novelty_and_store = fake_assess

            result = await submitter.submit_rigor_lean_theorem()
        finally:
            system_config.lean4_enabled = original_lean_enabled
            _restore_compiler_test_token_budgets(old_token_budgets)

        self.assertIsNotNone(result)
        self.assertEqual(result.source_id, "paper_456")
        complete_events = [data for event, data in events if event == "proof_check_complete"]
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0]["verified_count"], 1)
        self.assertEqual(complete_events[0]["novel_count"], 1)
        self.assertEqual(complete_events[0]["source_id"], "paper_456")
        self.assertEqual(complete_events[0]["novelty_tier"], "novel_variant")


if __name__ == "__main__":
    unittest.main()
