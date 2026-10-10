import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.installation_identity import InstallationBinding
from backend.shared.models import LocalNoveltyAssessmentProvenance, ProofRecord
from backend.shared.proof_locator import (
    CLIENT_BATCH_ID_VERSION,
    build_proof_locator,
    canonical_locator_digest,
    new_client_batch_id,
)


class CanonicalProofLocatorTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _record(proof_id: str = "proof_001") -> ProofRecord:
        return ProofRecord(
            proof_id=proof_id,
            theorem_statement="True",
            theorem_name="identity_true",
            source_type="paper",
            source_id="paper_identity",
            run_id="run-identity",
            lean_code="theorem identity_true : True := by trivial",
            novelty_tier="not_novel",
            independent_novelty_tier="not_novel",
            local_novelty_assessment_provenance=(
                LocalNoveltyAssessmentProvenance(
                    assessment_origin="caller_supplied",
                    source_local_rank="not_novel",
                )
            ),
        )

    async def test_same_local_id_in_distinct_stores_has_distinct_occurrence_identity(
        self,
    ) -> None:
        binding = InstallationBinding(
            installation_id="moto_installation_test",
            root_binding_id="moto_root_test",
        )
        with mock.patch(
            "backend.autonomous.memory.proof_database.ensure_installation_binding",
            return_value=binding,
        ), self.subTest("distinct stores"):
            import tempfile

            with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
                first_db = ProofDatabase()
                first_db.set_base_dir(Path(first_dir))
                await first_db.initialize()
                second_db = ProofDatabase()
                second_db.set_base_dir(Path(second_dir))
                await second_db.initialize()

                first = await first_db.add_proof_occurrence(self._record())
                second = await second_db.add_proof_occurrence(self._record())

        assert first.client_proof_id != second.client_proof_id
        assert first.canonical_locator.stable_store_id != second.canonical_locator.stable_store_id
        assert first.canonical_locator.proof_id == second.canonical_locator.proof_id == "proof_001"

    async def test_duplicate_local_id_is_rejected_instead_of_replaced(self) -> None:
        import tempfile

        binding = InstallationBinding(
            installation_id="moto_installation_test",
            root_binding_id="moto_root_test",
        )
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "backend.autonomous.memory.proof_database.ensure_installation_binding",
            return_value=binding,
        ):
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            original = await database.add_proof_occurrence(self._record())
            with self.assertRaisesRegex(ValueError, "Proof ID already exists"):
                await database.add_proof_occurrence(
                    self._record().model_copy(
                        update={
                            "theorem_statement": "False = False",
                            "lean_code": "theorem identity_false : False = False := rfl",
                        }
                    )
                )
            stored = await database.get_proof("proof_001")
            self.assertEqual(stored.client_proof_id, original.client_proof_id)

    async def test_manual_archive_preserves_store_locator_and_client_id(self) -> None:
        import tempfile

        binding = InstallationBinding(
            installation_id="moto_installation_test",
            root_binding_id="moto_root_test",
        )
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "backend.autonomous.memory.proof_database.ensure_installation_binding",
            return_value=binding,
        ):
            root = Path(tmpdir)
            active = root / "active"
            history = root / "history"
            database = ProofDatabase()
            database.set_base_dir(active)
            await database.initialize()
            stored = await database.add_proof_occurrence(self._record())
            run_id = await database.get_or_create_active_run_id()
            (history / run_id).mkdir(parents=True)

            metadata = await database.archive_current_run(history)
            archived = await database.get_library_proof_from_history(
                history,
                metadata["session_id"],
                stored.proof_id,
            )
            active_index = json.loads((active / "proofs_index.json").read_text(encoding="utf-8"))

        assert metadata["session_id"].endswith("_2")
        assert metadata["run_id"] == run_id
        assert archived["client_proof_id"] == stored.client_proof_id
        assert archived["canonical_locator"] == stored.canonical_locator.model_dump(mode="json")
        assert (
            archived["local_novelty_assessment_receipt"]
            == stored.local_novelty_assessment_receipt.model_dump(mode="json")
        )
        assert active_index["stable_store_id"] != stored.canonical_locator.stable_store_id

    def test_locator_digest_and_batch_id_are_stable_primitives(self) -> None:
        locator = build_proof_locator(
            binding=InstallationBinding(
                installation_id="moto_installation_test",
                root_binding_id="moto_root_test",
            ),
            store_kind="manual",
            stable_store_id="moto_store_test",
            run_id="run-test",
            proof_id="proof_001",
            canonical_identity_version="moto-proof-identity-v1",
            theorem_hash="a" * 64,
            lean_code_hash="b" * 64,
        )
        assert canonical_locator_digest(locator) == canonical_locator_digest(locator)
        first = new_client_batch_id()
        second = new_client_batch_id()
        assert first.startswith("moto_batch_")
        assert first != second
        assert CLIENT_BATCH_ID_VERSION == "moto-client-batch-id-v1"

    async def test_conflicting_record_store_ids_fail_closed_on_rebuild(self) -> None:
        import tempfile

        first_locator = build_proof_locator(
            binding=InstallationBinding(
                installation_id="moto_installation_test",
                root_binding_id="moto_root_test",
            ),
            store_kind="manual",
            stable_store_id="moto_store_first",
            run_id="run-test",
            proof_id="proof_001",
            canonical_identity_version="moto-proof-identity-v1",
            theorem_hash="a" * 64,
            lean_code_hash="b" * 64,
        )
        second_locator = first_locator.model_copy(
            update={
                "stable_store_id": "moto_store_second",
                "proof_id": "proof_002",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for proof_id, locator in (
                ("proof_001", first_locator),
                ("proof_002", second_locator),
            ):
                (root / f"proof_{proof_id}.json").write_text(
                    json.dumps(
                        self._record(proof_id).model_copy(
                            update={
                                "client_proof_id": f"moto_{proof_id}",
                                "canonical_locator": locator,
                                "canonical_locator_digest": canonical_locator_digest(locator),
                            }
                        ).model_dump(mode="json"),
                        default=str,
                    ),
                    encoding="utf-8",
                )
            (root / "proofs_index.json").write_text("{broken", encoding="utf-8")
            database = ProofDatabase()
            database.set_base_dir(root)
            with self.assertRaisesRegex(RuntimeError, "conflicting canonical stable store IDs"):
                await database.initialize()
