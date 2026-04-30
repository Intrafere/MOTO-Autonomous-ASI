import json
import unittest

from backend.compiler.validation import compiler_validator as validator_module
from backend.compiler.validation.compiler_validator import CompilerValidator
from backend.shared.models import CompilerSubmission


class LeanPlacementValidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_lean_placement_prompt_and_forced_rigor_check(self) -> None:
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
                                }
                            )
                        }
                    }
                ]
            }

        original_ensure = validator_module.paper_memory.ensure_markers_intact
        original_generate = validator_module.api_client_manager.generate_completion
        try:
            validator_module.paper_memory.ensure_markers_intact = fake_ensure_markers_intact
            validator_module.api_client_manager.generate_completion = fake_generate_completion

            result = await validator.validate_submission(
                submission,
                current_paper="Anchor paragraph.\n",
                current_outline="I. Outline",
            )
        finally:
            validator_module.paper_memory.ensure_markers_intact = original_ensure
            validator_module.api_client_manager.generate_completion = original_generate

        self.assertEqual(result.decision, "reject")
        self.assertTrue(result.rigor_check)
        self.assertFalse(result.placement_check)
        self.assertIn("Lean 4 Verified Theorem Placement", captured_prompt["text"])
        self.assertIn("MUST NOT re-evaluate", captured_prompt["text"])


if __name__ == "__main__":
    unittest.main()
