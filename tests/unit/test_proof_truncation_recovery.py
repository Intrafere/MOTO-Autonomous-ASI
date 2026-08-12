import unittest
from unittest.mock import AsyncMock, patch

from backend.autonomous.agents.proof_formalization_agent import (
    ProofFormalizationAgent,
    TRUNCATION_RECOVERY_POLICY_VERSION,
    _truncation_recovery_settings,
    _truncation_recovery_step,
)
from backend.autonomous.prompts.proof_prompts import (
    build_compact_proof_formalization_prompt,
    build_compact_proof_tactic_script_prompt,
)
from backend.shared.lean4_client import Lean4Result
from backend.shared.models import ProofAttemptFeedback, ProofCandidate


def _prompt_kwargs():
    return {
        "user_prompt": "Prove the requested target.",
        "source_type": "brainstorm",
        "theorem_statement": "True",
        "formal_sketch": "Use True.intro.",
        "full_source_content": "Mandatory complete source.",
        "source_excerpt": "Focused source.",
        "prior_attempts": [],
    }


class ProofTruncationRecoveryTests(unittest.TestCase):
    def test_configured_attempt_precedes_recovery(self):
        self.assertEqual(_truncation_recovery_step([]), 1)
        self.assertEqual(
            _truncation_recovery_settings(1),
            ("configured", None, "json"),
        )

    def test_recovery_progresses_by_truncation_count(self):
        attempts = []
        expected = [
            ("compact_same_model", None, "compact_json"),
            ("compact_reduced_reasoning", "low", "compact_json"),
            ("tactic_reduced_reasoning", "low", "compact_json"),
            ("tactic_minimal_reasoning", "none", "compact_json"),
        ]
        for index, settings in enumerate(expected, start=1):
            attempts.append(
                ProofAttemptFeedback(
                    attempt=index,
                    theorem_id="candidate",
                    failure_kind="output_truncated",
                    recovery_policy_version=TRUNCATION_RECOVERY_POLICY_VERSION,
                )
            )
            step = _truncation_recovery_step(attempts)
            self.assertEqual(_truncation_recovery_settings(step), settings)

    def test_non_truncation_does_not_erase_recovery_episode(self):
        attempts = [
            ProofAttemptFeedback(
                attempt=1,
                theorem_id="candidate",
                failure_kind="output_truncated",
            ),
            ProofAttemptFeedback(
                attempt=2,
                theorem_id="candidate",
                failure_kind="malformed_output",
            ),
        ]
        self.assertEqual(_truncation_recovery_step(attempts), 2)

    def test_compact_full_contract_is_terminal_and_minimal(self):
        prompt = build_compact_proof_formalization_prompt(**_prompt_kwargs())
        self.assertIn("Mandatory complete source.", prompt)
        self.assertIn('"lean_code"', prompt)
        self.assertNotIn('"reasoning": "brief note about the formalization strategy"', prompt)
        self.assertTrue(prompt.rstrip().endswith("}"))

    def test_compact_tactic_contract_uses_string_tactics(self):
        prompt = build_compact_proof_tactic_script_prompt(**_prompt_kwargs())
        self.assertIn("Mandatory complete source.", prompt)
        self.assertIn('"tactics": ["exact proof_term"]', prompt)
        self.assertNotIn('"reasoning": "Apply the core proof term', prompt)
        self.assertTrue(prompt.rstrip().endswith("}"))


class CappedValidResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_capped_full_script_runs_lean_and_preserves_rejection(self):
        agent = ProofFormalizationAgent("model", 8000, 1000, "proof-role")
        response = {
            "choices": [{
                "message": {
                    "content": '{"lean_code":"import Mathlib\\n\\ntheorem capped : True := by trivial"}'
                },
                "finish_reason": "length",
            }]
        }
        lean_client = AsyncMock()
        lean_client.check_proof.return_value = Lean4Result(
            success=False,
            error_output="type mismatch",
        )
        candidate = ProofCandidate(theorem_id="capped", statement="True")
        with (
            patch(
                "backend.autonomous.agents.proof_formalization_agent.api_client_manager.generate_completion",
                new=AsyncMock(return_value=response),
            ),
            patch(
                "backend.autonomous.agents.proof_formalization_agent.get_lean4_client",
                return_value=lean_client,
            ),
        ):
            _, _, feedback = await agent._run_full_script_attempt(
                user_research_prompt="Prove True.",
                source_type="paper",
                theorem_candidate=candidate,
                prior_attempts=[],
                source_excerpt="True",
                source_content="Complete source proving True.",
                attempt_number=1,
            )

        lean_client.check_proof.assert_awaited_once()
        self.assertTrue(feedback.lean_was_run)
        self.assertEqual(feedback.failure_kind, "lean_rejected")
        self.assertNotEqual(feedback.failure_kind, "output_truncated")


if __name__ == "__main__":
    unittest.main()
