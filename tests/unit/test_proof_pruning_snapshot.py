from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase

from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import ProofPruneCommitIntent, ProofRecord


class ProofPruningSnapshotTests(IsolatedAsyncioTestCase):
    async def test_snapshot_accounts_for_all_active_proofs_and_bounds_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            lean_code = "theorem duplicate_true : True := by trivial"
            for proof_id, source_id in (
                ("proof-old", "paper-old"),
                ("proof-new", "paper-new"),
            ):
                stored = await database.add_proof_occurrence(
                    ProofRecord(
                        proof_id=proof_id,
                        theorem_name="duplicate_true",
                        theorem_statement="True",
                        source_type="paper",
                        source_id=source_id,
                        run_id="owning-run",
                        lean_code=lean_code,
                        novel=True,
                        novelty_tier="mathematical_discovery",
                    )
                )
                await database.update_proof_dependencies(
                    stored.proof_id,
                    [],
                    extraction_status="complete",
                )

            snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="paper-old",
                canonical_user_prompt="Prove the objective.",
                trigger_reasons=["scheduled"],
            )

            self.assertEqual(len(snapshot.whole_set), 2)
            self.assertEqual(
                {entry.proof_id for entry in snapshot.whole_set},
                {"proof-old", "proof-new"},
            )
            self.assertEqual(len(snapshot.candidate_descriptors), 2)
            self.assertFalse(snapshot.evidence_bounded)
            self.assertTrue(all(entry.eligible_candidate for entry in snapshot.whole_set))

    async def test_snapshot_bounds_legacy_oversized_source_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            for proof_id in ("proof-old", "proof-new"):
                stored = await database.add_proof_occurrence(
                    ProofRecord(
                        proof_id=proof_id,
                        theorem_name="duplicate_true",
                        theorem_statement="True",
                        source_type="paper",
                        source_id=proof_id,
                        source_title="oversized title " * 200,
                        run_id="owning-run",
                        lean_code="theorem duplicate_true : True := by trivial",
                        novel=True,
                        novelty_tier="mathematical_discovery",
                    )
                )
                await database.update_proof_dependencies(
                    stored.proof_id,
                    [],
                    extraction_status="complete",
                )

            snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="proof-old",
                canonical_user_prompt="Prove the objective.",
            )

            self.assertTrue(snapshot.candidate_descriptors)
            self.assertTrue(
                all(
                    len(descriptor.source_title) == 1000
                    for descriptor in snapshot.candidate_descriptors
                )
            )

    async def test_dependency_updates_advance_revision_and_protect_unknown_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            stored = await database.add_proof_occurrence(
                ProofRecord(
                    proof_id="proof-one",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper-one",
                    run_id="owning-run",
                    lean_code="theorem one : True := by trivial",
                )
            )
            before = await database.get_proof_set_revision()
            updated = await database.update_proof_dependencies(
                stored.proof_id,
                [],
                extraction_status="complete",
            )
            after = await database.get_proof_set_revision()

            self.assertGreater(after, before)
            self.assertEqual(updated.dependency_extraction_status, "complete")

            snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="paper-one",
                canonical_user_prompt="Prove the objective.",
            )
            entry = snapshot.whole_set[0]
            self.assertNotIn("dependency_extraction_incomplete", entry.protected_reasons)
            self.assertTrue(entry.eligible_candidate)

    async def test_same_run_pruned_occurrence_is_not_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            stored = await database.add_proof_occurrence(
                ProofRecord(
                    proof_id="proof-pruned",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper-one",
                    run_id="owning-run",
                    lean_code="theorem pruned : True := by trivial",
                )
            )
            revision = await database.get_proof_set_revision()
            await database.set_live_context_status(
                proof_id=stored.proof_id,
                status="pruned",
                expected_run_id="owning-run",
                expected_proof_set_revision=revision,
                actor="automatic_proof_pruning",
                reason="Redundant exact occurrence.",
            )

            snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="paper-one",
                canonical_user_prompt="Prove the objective.",
            )
            self.assertEqual(snapshot.whole_set, [])

    async def test_commit_allows_unrelated_addition_but_rechecks_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            lean_code = "theorem duplicate_true : True := by trivial"
            stored_ids = []
            for proof_id in ("proof-old", "proof-new"):
                stored = await database.add_proof_occurrence(
                    ProofRecord(
                        proof_id=proof_id,
                        theorem_name="duplicate_true",
                        theorem_statement="True",
                        source_type="paper",
                        source_id=proof_id,
                        run_id="owning-run",
                        lean_code=lean_code,
                        novel=True,
                        novelty_tier="mathematical_discovery",
                    )
                )
                stored_ids.append(stored.proof_id)
                await database.update_proof_dependencies(
                    stored.proof_id,
                    [],
                    extraction_status="complete",
                )
            snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="paper-one",
                canonical_user_prompt="Prove the objective.",
            )
            target = snapshot.candidate_descriptors[0]
            intent = ProofPruneCommitIntent(
                snapshot_id=snapshot.snapshot_id,
                proof_id=target.proof_id,
                owning_run_id="owning-run",
                proof_set_revision=snapshot.proof_set_revision,
                expected_theorem_hash=target.canonical_theorem_hash,
                expected_lean_hash=target.canonical_lean_hash,
                prune_category="redundant",
                supporting_proof_ids=[
                    item.proof_id
                    for item in snapshot.candidate_descriptors
                    if item.proof_id != target.proof_id
                ][:1],
                supporting_proof_fingerprints={
                    item.proof_id: item.descriptor_fingerprint
                    for item in snapshot.candidate_descriptors
                    if item.proof_id != target.proof_id
                },
                target_dependency_fingerprint=target.dependency_fingerprint,
                target_descriptor_fingerprint=target.descriptor_fingerprint,
                evidence_policy_version=snapshot.evidence_policy_version,
                evidence_fingerprint=snapshot.evidence_fingerprint,
                trigger_reasons=["three_novel_proofs"],
                proposer_reasoning="This exact occurrence is redundant.",
                validator_reasoning="A stronger active exact occurrence remains.",
            )

            unrelated = await database.add_proof_occurrence(
                ProofRecord(
                    proof_id="unrelated",
                    theorem_name="unrelated",
                    theorem_statement="1 = 1",
                    source_type="paper",
                    source_id="paper-other",
                    run_id="owning-run",
                    lean_code="theorem unrelated : 1 = 1 := by rfl",
                    novel=True,
                    novelty_tier="mathematical_discovery",
                )
            )
            await database.update_proof_dependencies(
                unrelated.proof_id,
                [],
                extraction_status="complete",
            )
            updated, revision = await database.commit_pruning_intent(
                intent,
                snapshot=snapshot,
                expected_proof_store_id="manual:active",
                expected_proof_run_id="proof-run-1",
                expected_lifecycle_generation=1,
            )
            self.assertEqual(updated.live_context_status, "pruned")
            self.assertEqual(
                updated.live_context_prune_supporting_proof_ids,
                intent.supporting_proof_ids,
            )
            self.assertGreater(revision, snapshot.proof_set_revision)

            next_snapshot = await database.capture_pruning_snapshot(
                proof_store_id="manual:active",
                owning_run_id="owning-run",
                proof_run_id="proof-run-1",
                proof_run_lifecycle_generation=1,
                scope="manual",
                source_type="paper",
                source_id="paper-one",
                canonical_user_prompt="Prove the objective.",
            )
            support_entry = next(
                entry
                for entry in next_snapshot.whole_set
                if entry.proof_id == intent.supporting_proof_ids[0]
            )
            self.assertIn("retained_prune_support", support_entry.protected_reasons)
            self.assertFalse(support_entry.eligible_candidate)
