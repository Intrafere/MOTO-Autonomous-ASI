import asyncio
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, mock

from backend.api.routes import proofs as proofs_route
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import ProofRecord


class ProofStatusResponsivenessTests(IsolatedAsyncioTestCase):
    async def test_status_uses_cached_counts_without_initializing_stores(self) -> None:
        previous_lean_enabled = proofs_route.system_config.lean4_enabled
        previous_smt_enabled = proofs_route.system_config.smt_enabled
        proofs_route.system_config.lean4_enabled = False
        proofs_route.system_config.smt_enabled = False
        try:
            with (
                mock.patch.object(
                    proofs_route,
                    "_get_manual_check_status",
                    mock.AsyncMock(return_value=(False, "Lean 4 is disabled.")),
                ),
                mock.patch.object(
                    proofs_route.proof_database,
                    "count_proofs",
                    side_effect=AssertionError("status must not initialize the proof store"),
                ),
                mock.patch.object(
                    proofs_route.manual_proof_database,
                    "count_proofs",
                    side_effect=AssertionError("status must not initialize the manual proof store"),
                ),
                mock.patch.object(
                    proofs_route.proof_database,
                    "count_proofs_cached",
                    return_value={"total": 7},
                ),
                mock.patch.object(
                    proofs_route.manual_proof_database,
                    "count_proofs_cached",
                    return_value={"total": 3},
                ),
            ):
                payload = await proofs_route.get_proofs_status()
        finally:
            proofs_route.system_config.lean4_enabled = previous_lean_enabled
            proofs_route.system_config.smt_enabled = previous_smt_enabled

        self.assertEqual(payload["proof_counts"], {"total": 7})
        self.assertEqual(payload["manual_proof_counts"], {"total": 3})

    async def test_cleanup_traversal_runs_as_one_threaded_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            brainstorm_dir = root / "brainstorms"
            paper_dir = root / "papers"
            brainstorm_dir.mkdir()
            paper_dir.mkdir()
            brainstorm_path = brainstorm_dir / "brainstorm_topic.txt"
            brainstorm_path.write_text(
                "Body\n=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===\n\n"
                "Proof 1:\nStatus: Verified (Known)\nknown proof\n",
                encoding="utf-8",
            )

            calls = []

            async def run_in_thread(function, *args, **kwargs):
                calls.append((function, args, kwargs))
                return await asyncio.get_running_loop().run_in_executor(
                    None, lambda: function(*args, **kwargs)
                )

            with (
                mock.patch.object(proofs_route.brainstorm_memory, "_base_dir", brainstorm_dir),
                mock.patch.object(proofs_route.paper_library, "_base_dir", paper_dir),
                mock.patch.object(proofs_route.asyncio, "to_thread", side_effect=run_in_thread),
            ):
                result = await proofs_route._strip_known_proofs_from_files()
            cleaned_content = brainstorm_path.read_text(encoding="utf-8")

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["entries_removed"], 1)
        self.assertNotIn("Verified (Known)", cleaned_content)


class ProofPersistenceRollbackTests(IsolatedAsyncioTestCase):
    async def test_lean_write_failure_never_publishes_authoritative_json_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = ProofDatabase()
            database.set_base_dir(root)
            await database.initialize()
            original_writer = database._atomic_write_text_sync

            def fail_lean_write(path: Path, content: str) -> None:
                if path.suffix == ".lean":
                    raise OSError("simulated Lean write failure")
                original_writer(path, content)

            record = ProofRecord(
                proof_id="proof_rollback",
                theorem_statement="True",
                source_type="paper",
                source_id="paper_rollback",
                lean_code="theorem rollback_target : True := by trivial",
            )
            with mock.patch.object(database, "_atomic_write_text_sync", side_effect=fail_lean_write):
                with self.assertRaisesRegex(OSError, "simulated Lean write failure"):
                    await database.add_proof_occurrence(record)

            self.assertFalse((root / "proof_proof_rollback.json").exists())
            self.assertFalse((root / "proof_proof_rollback_lean.lean").exists())

            reconstructed = ProofDatabase()
            reconstructed.set_base_dir(root)
            await reconstructed.initialize()
            self.assertEqual((await reconstructed.get_all_proofs()), [])

