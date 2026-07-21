import unittest
from types import SimpleNamespace

from backend.autonomous.agents import proof_formalization_agent as proof_formalization_module
from backend.compiler.agents import high_param_submitter as high_param_module
from backend.compiler.agents.high_param_submitter import (
    HighParamSubmitter,
    _strip_generated_proofs_for_rigor_context,
)
from backend.compiler.memory.paper_memory import (
    APPENDIX_EMPTY_PLACEHOLDER,
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
)
from backend.compiler.prompts.rigor_prompts import build_rigor_theorem_discovery_prompt
from backend.shared.config import system_config
from backend.shared.models import ProofCandidate


class RigorPromptSourceContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_prompt_direct_injects_source_material_context(self):
        prompt = await build_rigor_theorem_discovery_prompt(
            user_prompt="Write a paper that advances the target problem.",
            current_outline="I. Body",
            current_paper="Current paper sentinel.",
            source_material_context="Source brainstorm sentinel with proof-relevant ideas.",
            source_material_label="Source brainstorm topic_001",
            existing_verified_proofs=[
                {
                    "proof_id": "proof_001",
                    "novel": True,
                    "theorem_statement": "Verified proof sentinel.",
                }
            ],
        )

        self.assertIn("SOURCE BRAINSTORM TOPIC_001", prompt)
        self.assertIn("Source brainstorm sentinel", prompt)
        self.assertIn("Verified proof sentinel", prompt)
        self.assertIn("Current paper sentinel", prompt)

    async def test_rigor_context_strips_generated_theorem_appendix_entries(self):
        paper = (
            "Current paper body sentinel.\n\n"
            f"{THEOREMS_APPENDIX_START}\n"
            "Theorem (proof_001) [Novel] - generated_appendix_duplicate\n"
            "Lean 4 proof:\n"
            "theorem generated_appendix_duplicate : True := by trivial\n"
            f"{THEOREMS_APPENDIX_END}\n"
        )

        stripped = _strip_generated_proofs_for_rigor_context(paper)

        self.assertIn("Current paper body sentinel.", stripped)
        self.assertIn(APPENDIX_EMPTY_PLACEHOLDER, stripped)
        self.assertNotIn("generated_appendix_duplicate", stripped)

    async def test_rigor_submitter_does_not_full_inject_proof_library_into_user_prompt(self):
        class FakeProofDatabase:
            def inject_into_prompt(self, prompt):
                return "FULL PROOF LIBRARY SHOULD NOT BE INJECTED\n" + prompt

        old_context = system_config.compiler_high_param_context_window
        old_output = system_config.compiler_high_param_max_output_tokens
        try:
            system_config.compiler_high_param_context_window = 8000
            system_config.compiler_high_param_max_output_tokens = 1000
            submitter = HighParamSubmitter(
                "model",
                "User prompt sentinel.",
                proof_database_store=FakeProofDatabase(),
            )
        finally:
            system_config.compiler_high_param_context_window = old_context
            system_config.compiler_high_param_max_output_tokens = old_output

        self.assertEqual(submitter.user_prompt, "User prompt sentinel.")
        self.assertNotIn("FULL PROOF LIBRARY", submitter.user_prompt)

    async def test_rigor_formalization_passes_paper_title_to_proof_search_context(self):
        old_context = system_config.compiler_high_param_context_window
        old_output = system_config.compiler_high_param_max_output_tokens
        original_get_paper = high_param_module.paper_memory.get_paper
        original_validate = high_param_module.validate_full_lean_proof_integrity
        original_prove = proof_formalization_module.ProofFormalizationAgent.prove_candidate
        captured = {}

        async def fake_get_paper():
            return "Current paper body sentinel."

        async def fake_validate(**_kwargs):
            return SimpleNamespace(
                valid=True,
                actual_theorem_statement="",
                actual_theorem_name="",
                category="aligned",
                reason="",
                downshift_reason="",
            )

        async def fake_prove_candidate(self, *args, **kwargs):
            captured["source_title"] = kwargs.get("source_title")
            return True, "rigor_title_test", "theorem rigor_title_test : True := by trivial", []

        try:
            system_config.compiler_high_param_context_window = 8000
            system_config.compiler_high_param_max_output_tokens = 1000
            high_param_module.paper_memory.get_paper = fake_get_paper
            high_param_module.validate_full_lean_proof_integrity = fake_validate
            proof_formalization_module.ProofFormalizationAgent.prove_candidate = fake_prove_candidate
            submitter = HighParamSubmitter(
                "model",
                "User prompt sentinel.",
                validator_context_window=4000,
                validator_max_tokens=500,
            )
            submitter.set_rigor_proof_source("paper_789", "Rigor Search Title")

            result = await submitter._step_formalize(
                candidate=ProofCandidate(
                    theorem_id="rigor_title_test",
                    statement="True is true.",
                    formal_sketch="By trivial.",
                ),
                theorem_statement="True is true.",
            )
        finally:
            proof_formalization_module.ProofFormalizationAgent.prove_candidate = original_prove
            high_param_module.validate_full_lean_proof_integrity = original_validate
            high_param_module.paper_memory.get_paper = original_get_paper
            system_config.compiler_high_param_context_window = old_context
            system_config.compiler_high_param_max_output_tokens = old_output

        self.assertIsNotNone(result)
        self.assertEqual(captured["source_title"], "Rigor Search Title")


if __name__ == "__main__":
    unittest.main()
