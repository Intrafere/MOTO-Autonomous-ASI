import unittest

from backend.compiler.prompts.rigor_prompts import build_rigor_theorem_discovery_prompt


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


if __name__ == "__main__":
    unittest.main()
