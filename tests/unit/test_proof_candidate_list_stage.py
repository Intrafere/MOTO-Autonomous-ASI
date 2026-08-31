import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.autonomous.core.proof_verification_stage import (
    ProofVerificationStage,
    _candidate_list_review_scope,
    _normalize_candidate_list_checkpoint,
)
from backend.shared.config import system_config
from backend.shared.models import (
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
)


def _candidate(theorem_id: str) -> ProofCandidate:
    return ProofCandidate(
        theorem_id=theorem_id,
        statement=f"Novel statement {theorem_id}.",
    )


def _validation(
    candidates: list[ProofCandidate],
    *,
    approved_count: int,
    feedback: str = "Regenerate the non-novel targets.",
) -> ProofCandidateListValidation:
    return ProofCandidateListValidation(
        results=[
            ProofCandidateNoveltyDecision(
                theorem_id=candidate.theorem_id,
                decision=(
                    "approve_novel"
                    if index < approved_count
                    else "reject_not_novel"
                ),
                reasoning=f"Independent decision for {candidate.theorem_id}.",
            )
            for index, candidate in enumerate(candidates)
        ],
        feedback=feedback,
    )


class ProofCandidateListStageTests(unittest.IsolatedAsyncioTestCase):
    def test_malformed_checkpoint_state_is_discarded_fail_closed(self):
        scope = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="manual",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 1,
            },
        )
        candidate = _candidate("thm-0")
        malformed_states = [
            {
                "status": "approved",
                "review_scope": scope,
                "generation_attempt": "invalid",
                "list_fingerprint": "",
                "proposed_candidates": [],
                "approved_candidate_ids": [],
                "semantic_rejections": [],
            },
            {
                "status": "approved",
                "review_scope": scope,
                "generation_attempt": 1,
                "list_fingerprint": "wrong",
                "proposed_candidates": [candidate.model_dump(mode="json")],
                "approved_candidate_ids": [candidate.theorem_id],
                "semantic_rejections": [],
                "validation": {},
            },
            {
                "status": "rejected",
                "review_scope": scope,
                "generation_attempt": 1,
                "list_fingerprint": "",
                "proposed_candidates": [],
                "approved_candidate_ids": [],
                "semantic_rejections": [{"invalid": True}],
            },
        ]
        for state in malformed_states:
            with self.subTest(state=state):
                self.assertEqual(
                    _normalize_candidate_list_checkpoint(
                        state,
                        expected_scope=scope,
                        source_type="paper",
                        source_id="paper-1",
                    ),
                    {},
                )

    async def _run_stage(
        self,
        *,
        candidates: list[ProofCandidate],
        should_stop,
        broadcast_fn,
        checkpoint_callback=None,
        checkpoint_candidate_list_state=None,
        proof_round_index=1,
        trigger="manual",
        run_id="run-1",
        proof_run_context=None,
        canonical_user_prompt="",
        checkpoint_processed_candidate_ids=None,
    ):
        stage = ProofVerificationStage()
        with patch.object(system_config, "lean4_enabled", True):
            return await stage.run(
                content="Mandatory source.",
                source_type="paper",
                source_id="paper-1",
                source_title="Paper",
                user_prompt="Solve the objective.",
                canonical_user_prompt=canonical_user_prompt,
                submitter_model="submitter",
                submitter_context=8000,
                submitter_max_tokens=1000,
                validator_model="validator",
                validator_context=8000,
                validator_max_tokens=1000,
                broadcast_fn=broadcast_fn,
                novel_proofs_db=object(),
                theorem_candidates=candidates,
                source_reserved=True,
                release_source_on_exit=False,
                source_reservation_token="reservation",
                should_stop=should_stop,
                checkpoint_callback=checkpoint_callback,
                checkpoint_candidate_list_state=checkpoint_candidate_list_state,
                checkpoint_processed_candidate_ids=checkpoint_processed_candidate_ids,
                proof_round_index=proof_round_index,
                trigger=trigger,
                run_id=run_id,
                proof_run_context=proof_run_context,
            )

    async def test_exact_threshold_forwards_only_approved_subset_before_phase_a(self):
        candidates = [_candidate(f"thm-{index}") for index in range(4)]
        events = []
        stopped = False

        async def broadcast(event, payload):
            nonlocal stopped
            events.append((event, payload))
            if event == "proof_check_candidates_found":
                stopped = True

        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=AsyncMock(return_value=_validation(candidates, approved_count=3)),
        ), patch.object(
            ProofVerificationStage,
            "_run_lean_pipeline_for_candidate",
            new=AsyncMock(side_effect=AssertionError("Phase A must not start after Stop.")),
        ):
            await self._run_stage(
                candidates=candidates,
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
            )

        accepted = next(payload for event, payload in events if event == "proof_candidate_list_review_accepted")
        found = next(payload for event, payload in events if event == "proof_check_candidates_found")
        self.assertEqual(accepted["proposed_count"], 4)
        self.assertEqual(accepted["approved_count"], 3)
        self.assertEqual(accepted["candidate_ids"], ["thm-0", "thm-1", "thm-2"])
        self.assertEqual(found["proposed_count"], 4)
        self.assertEqual(found["approved_count"], 3)
        self.assertEqual(len(found["theorems_preview"]), 3)

    async def test_validator_receives_canonical_prompt_without_private_history(self):
        candidate = _candidate("thm-0")
        stopped = False

        async def broadcast(event, _payload):
            nonlocal stopped
            if event == "proof_check_candidates_found":
                stopped = True

        validate = AsyncMock(return_value=_validation([candidate], approved_count=1))
        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ):
            await self._run_stage(
                candidates=[candidate],
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                canonical_user_prompt="Canonical objective only.",
            )

        self.assertEqual(
            validate.await_args.kwargs["user_prompt"],
            "Canonical objective only.",
        )

    async def test_semantic_regeneration_retains_only_latest_five_for_same_round(self):
        initial = [_candidate("thm-0")]
        events = []
        checkpoints = []
        stopped = False
        generation_calls = 0

        async def broadcast(event, payload):
            events.append((event, payload))

        async def identify(*args, **kwargs):
            nonlocal generation_calls, stopped
            generation_calls += 1
            if generation_calls == 6:
                stopped = True
            return True, [_candidate(f"thm-{generation_calls}")]

        async def save_checkpoint(payload):
            checkpoints.append(payload)

        async def reject(self, *, candidates, **kwargs):
            return _validation(
                candidates,
                approved_count=0,
                feedback=f"semantic rejection {candidates[0].theorem_id}",
            )

        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=reject,
        ), patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofIdentificationAgent.identify_candidates",
            new=identify,
        ):
            await self._run_stage(
                candidates=initial,
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                checkpoint_callback=save_checkpoint,
            )

        rejection_events = [
            payload
            for event, payload in events
            if event == "proof_candidate_list_review_rejected"
        ]
        self.assertEqual(len(rejection_events), 6)
        state = checkpoints[-1]["candidate_list_review"]
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(
            [item["generation_attempt"] for item in state["semantic_rejections"]],
            [2, 3, 4, 5, 6],
        )

    async def test_validator_failure_is_not_recorded_as_semantic_rejection(self):
        candidates = [_candidate("thm-0")]
        events = []
        checkpoints = []

        async def broadcast(event, payload):
            events.append((event, payload))

        async def save_checkpoint(payload):
            checkpoints.append(payload)

        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=AsyncMock(side_effect=RuntimeError("provider transport failed")),
        ):
            result = await self._run_stage(
                candidates=candidates,
                should_stop=lambda: False,
                broadcast_fn=broadcast,
                checkpoint_callback=save_checkpoint,
            )

        self.assertTrue(result.had_error)
        self.assertNotIn(
            "proof_candidate_list_review_rejected",
            [event for event, _payload in events],
        )
        self.assertEqual(
            checkpoints[-1]["candidate_list_review"]["semantic_rejections"],
            [],
        )
        interrupted = [
            payload
            for event, payload in events
            if event == "proof_candidate_list_review_interrupted"
        ]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["list_attempt"], 1)
        self.assertEqual(interrupted[0]["error_kind"], "contract_error")

    async def test_rejected_checkpoint_regenerates_before_revalidation(self):
        rejected = _candidate("rejected")
        replacement = _candidate("replacement")
        scope = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="manual",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 1,
            },
        )
        from backend.autonomous.core.proof_verification_stage import (
            _candidate_list_fingerprint,
        )

        rejected_state = {
            "status": "rejected",
            "review_scope": scope,
            "generation_attempt": 1,
            "list_fingerprint": _candidate_list_fingerprint(
                [rejected], "paper", "paper-1"
            ),
            "proposed_candidates": [rejected.model_dump(mode="json")],
            "approved_candidate_ids": [],
            "semantic_rejections": [
                {
                    "list_fingerprint": _candidate_list_fingerprint(
                        [rejected], "paper", "paper-1"
                    ),
                    "generation_attempt": 1,
                    "proposed_count": 1,
                    "approved_count": 0,
                    "rejected_candidate_ids": ["rejected"],
                    "feedback": "Replace the non-novel target.",
                }
            ],
        }
        events = []
        stopped = False

        async def broadcast(event, payload):
            nonlocal stopped
            events.append((event, payload))
            if event == "proof_check_candidates_found":
                stopped = True

        identify = AsyncMock(return_value=(True, [replacement]))
        validate = AsyncMock(
            return_value=_validation([replacement], approved_count=1)
        )
        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofIdentificationAgent.identify_candidates",
            new=identify,
        ), patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ):
            await self._run_stage(
                candidates=[rejected],
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                checkpoint_candidate_list_state=rejected_state,
                proof_run_context={
                    "proof_run_id": "proof-run",
                    "lifecycle_generation": 1,
                },
            )

        identify.assert_awaited_once()
        validate.assert_awaited_once()
        reviewed_candidates = validate.await_args.kwargs["candidates"]
        self.assertEqual(
            [candidate.theorem_id for candidate in reviewed_candidates],
            ["replacement"],
        )
        started = [
            payload
            for event, payload in events
            if event == "proof_candidate_list_review_started"
        ]
        self.assertEqual([payload["candidate_ids"] for payload in started], [["replacement"]])

    async def test_empty_rejected_checkpoint_regenerates_with_feedback_first(self):
        rejected = _candidate("rejected")
        replacement = _candidate("replacement")
        scope = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="manual",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 1,
            },
        )
        rejected_state = {
            "status": "rejected",
            "review_scope": scope,
            "generation_attempt": 1,
            "list_fingerprint": "",
            "proposed_candidates": [],
            "approved_candidate_ids": [],
            "semantic_rejections": [
                {
                    "list_fingerprint": "old-list",
                    "generation_attempt": 1,
                    "proposed_count": 1,
                    "approved_count": 0,
                    "rejected_candidate_ids": ["rejected"],
                    "feedback": "Replace the non-novel target.",
                }
            ],
        }
        stopped = False

        async def broadcast(event, _payload):
            nonlocal stopped
            if event == "proof_check_candidates_found":
                stopped = True

        identify = AsyncMock(return_value=(True, [replacement]))
        validate = AsyncMock(
            return_value=_validation([replacement], approved_count=1)
        )
        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofIdentificationAgent.identify_candidates",
            new=identify,
        ), patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ):
            await self._run_stage(
                candidates=[rejected],
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                checkpoint_candidate_list_state=rejected_state,
                proof_run_context={
                    "proof_run_id": "proof-run",
                    "lifecycle_generation": 1,
                },
            )

        identify.assert_awaited_once()
        self.assertIn(
            "Replace the non-novel target.",
            identify.await_args.kwargs["candidate_list_rejection_feedback"],
        )

    async def test_cancelled_list_review_emits_interrupted_activity(self):
        candidate = _candidate("thm-0")
        events = []

        async def broadcast(event, payload):
            events.append((event, payload))

        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self._run_stage(
                    candidates=[candidate],
                    should_stop=lambda: False,
                    broadcast_fn=broadcast,
                )

        interrupted = [
            payload
            for event, payload in events
            if event == "proof_candidate_list_review_interrupted"
        ]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["error_kind"], "cancelled")

    async def test_approved_checkpoint_reuse_is_fenced_by_round_scope(self):
        candidate = _candidate("thm-0")
        original_scope = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="manual",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 4,
            },
        )
        stale_state = {
            "status": "approved",
            "review_scope": original_scope,
            "list_fingerprint": "stale",
            "generation_attempt": 9,
            "proposed_candidates": [candidate.model_dump(mode="json")],
            "approved_candidate_ids": [candidate.theorem_id],
            "semantic_rejections": [
                {
                    "list_fingerprint": "old",
                    "generation_attempt": 8,
                    "proposed_count": 1,
                    "approved_count": 0,
                    "rejected_candidate_ids": [candidate.theorem_id],
                    "feedback": "Old-round feedback.",
                }
            ],
        }
        stopped = False
        events = []

        async def broadcast(event, payload):
            nonlocal stopped
            events.append((event, payload))
            if event == "proof_check_candidates_found":
                stopped = True

        validate = AsyncMock(return_value=_validation([candidate], approved_count=1))
        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ):
            await self._run_stage(
                candidates=[candidate],
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                checkpoint_candidate_list_state=stale_state,
                proof_round_index=2,
                proof_run_context={
                    "proof_run_id": "proof-run",
                    "lifecycle_generation": 4,
                },
            )

        validate.assert_awaited_once()
        started = next(payload for event, payload in events if event == "proof_candidate_list_review_started")
        self.assertEqual(started["list_attempt"], 1)

    async def test_approved_checkpoint_reuses_ordered_remaining_projection_after_resume(self):
        candidates = [_candidate(f"thm-{index}") for index in range(4)]
        approved = candidates[:3]
        scope = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="manual",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 1,
            },
        )
        from backend.autonomous.core.proof_verification_stage import (
            _candidate_list_fingerprint,
        )

        validation = _validation(candidates, approved_count=3)
        state = {
            "status": "approved",
            "review_scope": scope,
            "generation_attempt": 1,
            "list_fingerprint": _candidate_list_fingerprint(
                candidates, "paper", "paper-1"
            ),
            "proposed_candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
            "approved_candidate_ids": [
                candidate.theorem_id for candidate in approved
            ],
            "semantic_rejections": [],
            "validation": validation.model_dump(mode="json"),
        }
        stopped = False
        events = []

        async def broadcast(event, payload):
            nonlocal stopped
            events.append((event, payload))
            if event == "proof_check_candidates_found":
                stopped = True

        validate = AsyncMock(side_effect=AssertionError("Approved list must not be re-reviewed."))
        with patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ):
            await self._run_stage(
                candidates=approved[1:],
                should_stop=lambda: stopped,
                broadcast_fn=broadcast,
                checkpoint_candidate_list_state=state,
                checkpoint_processed_candidate_ids=[approved[0].theorem_id],
                proof_run_context={
                    "proof_run_id": "proof-run",
                    "lifecycle_generation": 2,
                },
            )

        validate.assert_not_awaited()
        found = next(
            payload
            for event, payload in events
            if event == "proof_check_candidates_found"
        )
        self.assertEqual(found["proposed_count"], 4)
        self.assertEqual(found["approved_count"], 2)

    def test_review_scope_survives_lifecycle_resume_but_not_round_change(self):
        first = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="automatic",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 1,
            },
        )
        resumed = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="automatic",
            proof_round_index=1,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 2,
            },
        )
        next_round = _candidate_list_review_scope(
            source_type="paper",
            source_id="paper-1",
            run_id="run-1",
            trigger="automatic_round_2",
            proof_round_index=2,
            proof_run_context={
                "proof_run_id": "proof-run",
                "lifecycle_generation": 2,
            },
        )

        self.assertEqual(first, resumed)
        self.assertNotEqual(first, next_round)


if __name__ == "__main__":
    unittest.main()
