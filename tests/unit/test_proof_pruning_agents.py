from datetime import datetime
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from pydantic import ValidationError

from backend.autonomous.agents.proof_pruning_agent import (
    ProofPruningProposerAgent,
    ProofPruningReviewService,
    parse_proof_prune_proposal,
    parse_proof_prune_validation,
    proof_run_role_suffix,
    validate_proposal_against_snapshot,
)
from backend.shared.models import (
    ProofPruneAggregateEntry,
    ProofPruneProofDescriptor,
    ProofPruneProposal,
    ProofPruneSnapshot,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
)


def _snapshot() -> ProofPruneSnapshot:
    aggregate = [
        ProofPruneAggregateEntry(
            proof_id="proof-old",
            theorem_name="t",
            canonical_theorem_hash="th",
            canonical_lean_hash="lh",
            novelty_tier="mathematical_discovery",
            source_type="paper",
            source_id="paper-1",
            created_at=datetime(2026, 1, 1),
            dependency_extraction_status="complete",
            dependency_fingerprint="dep-old",
            descriptor_fingerprint="desc-old",
            eligible_candidate=True,
        ),
        ProofPruneAggregateEntry(
            proof_id="proof-new",
            theorem_name="t",
            canonical_theorem_hash="th",
            canonical_lean_hash="lh",
            novelty_tier="mathematical_discovery",
            source_type="paper",
            source_id="paper-2",
            created_at=datetime(2026, 1, 2),
            dependency_extraction_status="complete",
            dependency_fingerprint="dep-new",
            descriptor_fingerprint="desc-new",
            eligible_candidate=True,
        ),
    ]
    descriptors = [
        ProofPruneProofDescriptor(
            proof_id=entry.proof_id,
            theorem_name=entry.theorem_name,
            theorem_statement="True",
            canonical_theorem_hash=entry.canonical_theorem_hash,
            canonical_lean_hash=entry.canonical_lean_hash,
            novelty_tier=entry.novelty_tier,
            source_type=entry.source_type,
            source_id=entry.source_id,
            created_at=entry.created_at,
            dependency_extraction_status="complete",
            dependency_fingerprint=entry.dependency_fingerprint,
            descriptor_fingerprint=entry.descriptor_fingerprint,
            lean_code="theorem t : True := by trivial",
            lean_code_included=True,
        )
        for entry in aggregate
    ]
    return ProofPruneSnapshot(
        snapshot_id="snapshot",
        proof_set_revision=2,
        proof_store_id="manual:active",
        owning_run_id="owning-run",
        proof_run_id="proof-run-1",
        proof_run_lifecycle_generation=1,
        scope="manual",
        source_type="paper",
        source_id="paper-1",
        canonical_user_prompt="Prove the objective.",
        trigger_reasons=["scheduled"],
        whole_set=aggregate,
        candidate_descriptors=descriptors,
    )


def _runtime() -> ProofRuntimeConfigSnapshot:
    proposer = ProofRoleConfigSnapshot(
        provider="openrouter",
        model_id="proposer-model",
        openrouter_provider="Provider A",
        openrouter_reasoning_effort="high",
        lm_studio_fallback_id="fallback",
        context_window=32000,
        max_output_tokens=2000,
        supercharge_enabled=True,
    )
    validator = ProofRoleConfigSnapshot(
        provider="lm_studio",
        model_id="validator-model",
        context_window=24000,
        max_output_tokens=1500,
    )
    return ProofRuntimeConfigSnapshot(
        brainstorm=proposer,
        paper=proposer,
        validator=validator,
    )


class ProofPruningContractTests(TestCase):
    def test_no_prune_requires_null_targets(self) -> None:
        result = parse_proof_prune_proposal(
            '{"action":"no_prune","proof_id":null,'
            '"expected_theorem_hash":null,"expected_lean_hash":null,'
            '"reasoning":"All routes are unique."}'
        )
        self.assertEqual(result.action, "no_prune")
        with self.assertRaises(ValidationError):
            ProofPruneProposal(
                action="no_prune",
                proof_id="proof-old",
                expected_theorem_hash=None,
                expected_lean_hash=None,
                reasoning="invalid",
            )

    def test_proposal_requires_id_and_hashes(self) -> None:
        with self.assertRaises(ValidationError):
            ProofPruneProposal(
                action="propose_prune",
                proof_id="proof-old",
                expected_theorem_hash="th",
                expected_lean_hash=None,
                reasoning="missing Lean hash",
            )

    def test_validator_cannot_replace_target(self) -> None:
        with self.assertRaises(ValueError):
            parse_proof_prune_validation(
                '{"decision":"accept","proof_id":"other","reasoning":"ok"}',
                expected_proof_id="proof-old",
            )

    def test_guard_checks_hashes_and_dependency_state(self) -> None:
        snapshot = _snapshot()
        allowed = validate_proposal_against_snapshot(
            ProofPruneProposal(
                action="propose_prune",
                proof_id="proof-old",
                expected_theorem_hash="th",
                expected_lean_hash="lh",
                prune_category="superseded",
                supporting_proof_ids=["proof-new"],
                coverage_claims=[{
                    "target_contribution": "Truth result",
                    "preserved_by_proof_ids": ["proof-new"],
                    "explanation": "The retained proof establishes the stronger result.",
                }],
                reasoning="Semantically superseded occurrence.",
            ),
            snapshot,
        )
        self.assertTrue(allowed.allowed)
        rejected = validate_proposal_against_snapshot(
            ProofPruneProposal(
                action="propose_prune",
                proof_id="proof-old",
                expected_theorem_hash="wrong",
                expected_lean_hash="lh",
                prune_category="superseded",
                supporting_proof_ids=["proof-new"],
                coverage_claims=[{
                    "target_contribution": "Truth result",
                    "preserved_by_proof_ids": ["proof-new"],
                    "explanation": "The retained proof establishes it.",
                }],
                reasoning="Wrong identity.",
            ),
            snapshot,
        )
        self.assertFalse(rejected.allowed)
        self.assertIn("theorem_hash_mismatch", rejected.reasons)

    def test_role_suffix_is_stable_and_run_specific(self) -> None:
        first = proof_run_role_suffix("manual", "proof-run-1")
        self.assertEqual(first, proof_run_role_suffix("manual", "proof-run-1"))
        self.assertNotEqual(first, proof_run_role_suffix("manual", "proof-run-2"))


class ProofPruningAgentTests(IsolatedAsyncioTestCase):
    async def test_oversized_proposer_review_partitions_oldest_first(self) -> None:
        snapshot = _snapshot()
        snapshot.candidate_descriptors = [
            *snapshot.candidate_descriptors,
            *[
                snapshot.candidate_descriptors[-1].model_copy(
                    update={
                        "proof_id": f"proof-new-{index}",
                        "descriptor_fingerprint": f"desc-new-{index}",
                        "created_at": datetime(2026, 1, 3 + index),
                        "theorem_statement": "True " * 1000,
                    }
                )
                for index in range(3)
            ],
        ]
        agent = ProofPruningProposerAgent(
            role_id="test_proposer",
            task_prefix="proof_prune_propose",
            role_config=_runtime().paper,
        )
        seen_sections = []

        async def fake_generate(*, prompt, **_kwargs):
            seen_sections.append(prompt)
            return ProofPruneProposal(
                action="no_prune",
                proof_id=None,
                expected_theorem_hash=None,
                expected_lean_hash=None,
                reasoning="No removable proof in this section.",
            )

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent."
            "rag_config.__class__.get_available_input_tokens",
            return_value=5000,
        ), mock.patch.object(
            agent,
            "_generate_parse_with_one_repair",
            side_effect=fake_generate,
        ):
            result = await agent.propose(snapshot)

        self.assertEqual(result.action, "no_prune")
        self.assertGreater(len(seen_sections), 1)
        self.assertIn("proof-old", seen_sections[0])
        self.assertIn("proof-new", seen_sections[0])
        self.assertIn("proof-new-2", seen_sections[-1])
        self.assertIn("proof-old", seen_sections[-1])

    async def test_multiple_section_proposals_require_global_arbitration(self) -> None:
        snapshot = _snapshot()
        agent = ProofPruningProposerAgent(
            role_id="test_proposer_arbitration",
            task_prefix="proof_prune_propose",
            role_config=_runtime().paper,
        )
        calls = 0
        proposals = []
        for proof_id, support_id in (
            ("proof-old", "proof-new"),
            ("proof-new", "proof-old"),
        ):
            proposals.append(ProofPruneProposal(
                action="propose_prune",
                proof_id=proof_id,
                expected_theorem_hash="th",
                expected_lean_hash="lh",
                prune_category="redundant",
                supporting_proof_ids=[support_id],
                coverage_claims=[{
                    "target_contribution": "Truth result",
                    "preserved_by_proof_ids": [support_id],
                    "explanation": "The retained proof preserves the result.",
                }],
                reasoning="This section nominates one occurrence.",
            ))

        async def fake_generate(*, prompt, **_kwargs):
            nonlocal calls
            calls += 1
            return ProofPruneProposal(
                action="no_prune",
                proof_id=None,
                expected_theorem_hash=None,
                expected_lean_hash=None,
                reasoning="The section proposals conflict.",
            )

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent."
            "rag_config.__class__.get_available_input_tokens",
            return_value=6000,
        ), mock.patch.object(
            agent,
            "_generate_parse_with_one_repair",
            side_effect=fake_generate,
        ):
            result = await agent._arbitrate_section_proposals(snapshot, proposals)

        self.assertEqual(result.action, "no_prune")
        self.assertEqual(calls, 1)

    async def test_no_prune_skips_validator_and_preserves_role_configs(self) -> None:
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"action":"no_prune","proof_id":null,'
                                '"expected_theorem_hash":null,'
                                '"expected_lean_hash":null,'
                                '"reasoning":"Every proof remains useful."}'
                            )
                        }
                    }
                ]
            }

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.generate_completion",
            side_effect=fake_completion,
        ), mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.configure_role"
        ) as configure:
            service = ProofPruningReviewService(
                runtime_snapshot=_runtime(),
                scope="manual",
                proof_run_id="proof-run-1",
            )
            result = await service.review(_snapshot())

        self.assertEqual(result.outcome, "no_prune")
        self.assertEqual(len(calls), 1)
        configured = {call.args[0]: call.args[1] for call in configure.call_args_list}
        proposer_id = next(key for key in configured if "prune_proposer" in key)
        validator_id = next(key for key in configured if "prune_validator" in key)
        self.assertEqual(configured[proposer_id].model_id, "proposer-model")
        self.assertTrue(configured[proposer_id].supercharge_enabled)
        self.assertEqual(configured[validator_id].model_id, "validator-model")

    async def test_accept_returns_commit_intent_without_mutation(self) -> None:
        outputs = [
            (
                '{"action":"propose_prune","proof_id":"proof-old",'
                '"expected_theorem_hash":"th","expected_lean_hash":"lh",'
                '"prune_category":"superseded",'
                '"supporting_proof_ids":["proof-new"],'
                '"coverage_claims":[{"target_contribution":"Truth result",'
                '"preserved_by_proof_ids":["proof-new"],'
                '"explanation":"The retained proof establishes the stronger result."}],'
                '"reasoning":"The newer proof preserves the contribution."}'
            ),
            (
                '{"decision":"accept","proof_id":"proof-old",'
                '"supporting_proof_ids":["proof-new"],'
                '"coverage_confirmed":true,'
                '"reasoning":"No dependency or route is lost."}'
            ),
        ]

        async def fake_completion(**_kwargs):
            return {"choices": [{"message": {"content": outputs.pop(0)}}]}

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.generate_completion",
            side_effect=fake_completion,
        ), mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.configure_role"
        ):
            result = await ProofPruningReviewService(
                runtime_snapshot=_runtime(),
                scope="manual",
                proof_run_id="proof-run-1",
            ).review(_snapshot(), current_revision=2)

        self.assertEqual(result.outcome, "commit_intent")
        self.assertEqual(result.commit_intent.proof_id, "proof-old")
        self.assertEqual(result.commit_intent.proof_set_revision, 2)

    async def test_deterministic_guard_rejection_skips_validator(self) -> None:
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"action":"propose_prune","proof_id":"proof-old",'
                                '"expected_theorem_hash":"wrong",'
                                '"expected_lean_hash":"lh",'
                                '"prune_category":"superseded",'
                                '"supporting_proof_ids":["proof-new"],'
                                '"coverage_claims":[{"target_contribution":"Truth result",'
                                '"preserved_by_proof_ids":["proof-new"],'
                                '"explanation":"The retained proof establishes it."}],'
                                '"reasoning":"Attempt stale target."}'
                            )
                        }
                    }
                ]
            }

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.generate_completion",
            side_effect=fake_completion,
        ), mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.configure_role"
        ):
            result = await ProofPruningReviewService(
                runtime_snapshot=_runtime(),
                scope="manual",
                proof_run_id="proof-run-1",
            ).review(_snapshot())

        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(result.validation.decision, "reject")
        self.assertIn("theorem_hash_mismatch", result.validation.reasoning)
        self.assertEqual(len(calls), 1)

    async def test_malformed_output_gets_one_sanitized_repair(self) -> None:
        calls = []
        outputs = [
            "<think>private</think>{broken",
            (
                '{"action":"no_prune","proof_id":null,'
                '"expected_theorem_hash":null,"expected_lean_hash":null,'
                '"reasoning":"Insufficient evidence."}'
            ),
        ]

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": outputs.pop(0)}}]}

        with mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.generate_completion",
            side_effect=fake_completion,
        ), mock.patch(
            "backend.autonomous.agents.proof_pruning_agent.api_client_manager.configure_role"
        ):
            result = await ProofPruningReviewService(
                runtime_snapshot=_runtime(),
                scope="manual",
                proof_run_id="proof-run-1",
            ).review(_snapshot())

        self.assertEqual(result.outcome, "no_prune")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["task_id"].endswith("_retry"))
        retry_messages = calls[1]["messages"]
        assistant_messages = [
            message["content"]
            for message in retry_messages
            if message["role"] == "assistant"
        ]
        if assistant_messages:
            self.assertNotIn("<think>", assistant_messages[0])
