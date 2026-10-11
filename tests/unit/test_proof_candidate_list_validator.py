import json
import unittest
from unittest.mock import AsyncMock, patch

from backend.autonomous.agents.proof_candidate_list_validator import (
    ProofCandidateListContractError,
    ProofCandidateListValidator,
)
from backend.autonomous.prompts.proof_prompts import (
    build_proof_candidate_list_validation_prompt,
)
from backend.shared.models import ProofCandidate


class ProofCandidateListValidatorContractTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            ProofCandidate(theorem_id=f"thm_{index}", statement=f"Statement {index}")
            for index in range(1, 5)
        ]

    def test_prompt_approves_novel_indirect_steps_and_rejects_known_results(self):
        prompt = build_proof_candidate_list_validation_prompt(
            user_prompt="Solve the problem.",
            source_type="brainstorm",
            source_id="topic_001",
            source_title="Topic",
            candidates_json="[]",
        )

        self.assertIn("may eventually lead, directly or indirectly", prompt)
        self.assertIn("Novel intermediate or stepping-stone results qualify", prompt)
        self.assertIn("Reject standard results, routine helpers", prompt)
        self.assertNotIn("directly solves or substantially advances", prompt)

    def test_single_approved_candidate_accepts_the_list(self):
        self.assertTrue(ProofCandidateListValidator.list_accepted(approved_count=1))

    def test_zero_approved_candidates_rejects_the_list(self):
        self.assertFalse(ProofCandidateListValidator.list_accepted(approved_count=0))

    def test_partially_novel_list_is_accepted_and_only_approved_candidates_forward(self):
        validation = ProofCandidateListValidator.parse_response(
            json.dumps(
                {
                    "results": [
                        {
                            "theorem_id": candidate.theorem_id,
                            "decision": (
                                "approve_novel"
                                if candidate.theorem_id == "thm_2"
                                else "reject_not_novel"
                            ),
                            "reasoning": "Independent novelty assessment.",
                        }
                        for candidate in self.candidates
                    ],
                    "feedback": "Only the second target is novel.",
                }
            ),
            expected_candidate_ids=[candidate.theorem_id for candidate in self.candidates],
        )
        approved = ProofCandidateListValidator.approved_candidates(
            self.candidates,
            validation,
        )
        self.assertEqual(
            [candidate.theorem_id for candidate in approved],
            ["thm_2"],
        )
        self.assertTrue(
            ProofCandidateListValidator.list_accepted(approved_count=len(approved))
        )

    def test_duplicate_missing_or_reordered_results_are_rejected(self):
        bad_payloads = [
            ["thm_1", "thm_1", "thm_3", "thm_4"],
            ["thm_1", "thm_2", "thm_3"],
            ["thm_2", "thm_1", "thm_3", "thm_4"],
        ]
        for theorem_ids in bad_payloads:
            with self.subTest(theorem_ids=theorem_ids):
                with self.assertRaises(ValueError):
                    ProofCandidateListValidator.parse_response(
                        json.dumps(
                            {
                                "results": [
                                    {
                                        "theorem_id": theorem_id,
                                        "decision": "approve_novel",
                                        "reasoning": "Reason.",
                                    }
                                    for theorem_id in theorem_ids
                                ],
                                "feedback": "Feedback.",
                            }
                        ),
                        expected_candidate_ids=[
                            candidate.theorem_id for candidate in self.candidates
                        ],
                    )

    def test_malformed_extra_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            ProofCandidateListValidator.parse_response(
                json.dumps(
                    {
                        "results": [
                            {
                                "theorem_id": candidate.theorem_id,
                                "decision": "approve_novel",
                                "reasoning": "Reason.",
                                "unexpected": True,
                            }
                            for candidate in self.candidates
                        ],
                        "feedback": "Feedback.",
                    }
                ),
                expected_candidate_ids=[
                    candidate.theorem_id for candidate in self.candidates
                ],
            )


class ProofCandidateListValidatorAsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.candidates = [
            ProofCandidate(theorem_id="thm_1", statement="Statement 1"),
            ProofCandidate(theorem_id="thm_2", statement="Statement 2"),
        ]
        self.validator = ProofCandidateListValidator(
            model_id="validator-model",
            context_window=8000,
            max_output_tokens=1000,
            role_id="candidate-list-role",
        )

    @staticmethod
    def _response(payload):
        return {
            "choices": [
                {"message": {"content": json.dumps(payload)}}
            ]
        }

    async def test_validate_routes_exact_role_and_returns_valid_response(self):
        payload = {
            "results": [
                {
                    "theorem_id": candidate.theorem_id,
                    "decision": "approve_novel",
                    "reasoning": "Novel.",
                }
                for candidate in self.candidates
            ],
            "feedback": "Approved.",
        }
        generate = AsyncMock(return_value=self._response(payload))
        with patch(
            "backend.autonomous.agents.proof_candidate_list_validator."
            "api_client_manager.generate_completion",
            new=generate,
        ):
            validation = await self.validator.validate(
                user_prompt="Solve.",
                source_type="paper",
                source_id="paper-1",
                source_title="Paper",
                candidates=self.candidates,
            )

        self.assertEqual(len(validation.results), 2)
        self.assertEqual(generate.await_args.kwargs["role_id"], "candidate-list-role")
        self.assertEqual(generate.await_args.kwargs["task_id"], "proof_list_val_001")

    async def test_validate_uses_one_bounded_repair(self):
        repaired = {
            "results": [
                {
                    "theorem_id": candidate.theorem_id,
                    "decision": "approve_novel",
                    "reasoning": "Novel.",
                }
                for candidate in self.candidates
            ],
            "feedback": "Repaired.",
        }
        generate = AsyncMock(
            side_effect=[
                self._response({"wrong": True}),
                self._response(repaired),
            ]
        )
        with patch(
            "backend.autonomous.agents.proof_candidate_list_validator."
            "api_client_manager.generate_completion",
            new=generate,
        ):
            validation = await self.validator.validate(
                user_prompt="Solve.",
                source_type="paper",
                source_id="paper-1",
                source_title="Paper",
                candidates=self.candidates,
            )

        self.assertEqual(validation.feedback, "Repaired.")
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(
            generate.await_args_list[1].kwargs["task_id"],
            "proof_list_val_001_repair",
        )

    async def test_validate_repair_exhaustion_raises_contract_error(self):
        generate = AsyncMock(
            side_effect=[
                self._response({"wrong": True}),
                self._response({"still_wrong": True}),
            ]
        )
        with patch(
            "backend.autonomous.agents.proof_candidate_list_validator."
            "api_client_manager.generate_completion",
            new=generate,
        ):
            with self.assertRaises(ProofCandidateListContractError):
                await self.validator.validate(
                    user_prompt="Solve.",
                    source_type="paper",
                    source_id="paper-1",
                    source_title="Paper",
                    candidates=self.candidates,
                )

        self.assertEqual(generate.await_count, 2)

    async def test_provider_failure_propagates_without_repair(self):
        generate = AsyncMock(side_effect=RuntimeError("transport failed"))
        with patch(
            "backend.autonomous.agents.proof_candidate_list_validator."
            "api_client_manager.generate_completion",
            new=generate,
        ):
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                await self.validator.validate(
                    user_prompt="Solve.",
                    source_type="paper",
                    source_id="paper-1",
                    source_title="Paper",
                    candidates=self.candidates,
                )

        self.assertEqual(generate.await_count, 1)


if __name__ == "__main__":
    unittest.main()
