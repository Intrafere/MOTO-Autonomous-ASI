import asyncio
import json
from pathlib import Path
import tempfile
from unittest import TestCase, mock

from fastapi import HTTPException

from backend.aggregator.core.coordinator import _resolve_uploaded_user_file
from backend.aggregator.ingestion.pipeline import IngestionPipeline
from backend.api.routes.aggregator import _clear_uploaded_files, _delete_uploaded_file
from backend.api.routes import autonomous as autonomous_route
from backend.autonomous.memory.autonomous_rejection_logs import AutonomousRejectionLogs
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.config import system_config
from backend.shared.models import PaperMetadata, ProofRecord
from backend.shared.path_safety import resolve_filename_within_root


class IngestionPathHardeningTests(TestCase):
    def test_ingest_file_rejects_paths_outside_trusted_roots(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                trusted = root / "trusted"
                outside = root / "outside.txt"
                trusted.mkdir()
                outside.write_text("outside", encoding="utf-8")

                with self.assertRaises(ValueError):
                    await IngestionPipeline().ingest_file(
                        str(outside),
                        chunk_sizes=[256],
                        trusted_roots=[trusted],
                    )

        asyncio.run(run_case())

    def test_ingest_file_accepts_paths_inside_trusted_roots(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                trusted = root / "trusted"
                trusted.mkdir()
                paper = trusted / "paper.txt"
                paper.write_text("A useful mathematical note.", encoding="utf-8")

                chunks = await IngestionPipeline().ingest_file(
                    str(paper),
                    chunk_sizes=[256],
                    trusted_roots=[trusted],
                )

            self.assertIn(256, chunks)
            self.assertGreaterEqual(len(chunks[256]), 1)

        asyncio.run(run_case())


class UploadPathResolutionTests(TestCase):
    def test_uploaded_user_file_rejects_traversal_and_untrusted_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploads = root / "uploads"
            data = root / "data"
            outside = root / "outside.txt"
            uploads.mkdir()
            data.mkdir()
            outside.write_text("outside", encoding="utf-8")

            with mock.patch.object(system_config, "user_uploads_dir", str(uploads)):
                with mock.patch.object(system_config, "data_dir", str(data)):
                    self.assertIsNone(_resolve_uploaded_user_file("../outside.txt"))
                    self.assertIsNone(_resolve_uploaded_user_file(str(outside)))

    def test_uploaded_user_file_allows_trusted_context_files_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploads = root / "uploads"
            data = root / "data"
            trusted_file = data / "paper.txt"
            uploads.mkdir()
            data.mkdir()
            trusted_file.write_text("trusted", encoding="utf-8")

            with mock.patch.object(system_config, "user_uploads_dir", str(uploads)):
                with mock.patch.object(system_config, "data_dir", str(data)):
                    self.assertIsNone(
                        _resolve_uploaded_user_file(
                            str(trusted_file),
                            allow_trusted_context_files=False,
                        )
                    )
                    self.assertEqual(
                        _resolve_uploaded_user_file(
                            str(trusted_file),
                            allow_trusted_context_files=True,
                        ),
                        trusted_file.resolve(),
                    )

    def test_delete_uploaded_file_rejects_traversal_and_deletes_lean(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                uploads = Path(temp_dir) / "uploads"
                uploads.mkdir()
                lean_file = uploads / "helper.lean"
                lean_file.write_text("theorem helper : True := by trivial", encoding="utf-8")

                with mock.patch.object(system_config, "user_uploads_dir", str(uploads)):
                    self.assertTrue(await _delete_uploaded_file("helper.lean"))
                    self.assertFalse(lean_file.exists())
                    with self.assertRaises(ValueError):
                        await _delete_uploaded_file("../outside.lean")
                    with self.assertRaises(ValueError):
                        await _delete_uploaded_file("not_allowed.md")

        asyncio.run(run_case())

    def test_clear_uploaded_files_only_deletes_managed_text_uploads(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                uploads = Path(temp_dir) / "uploads"
                uploads.mkdir()
                (uploads / "notes.txt").write_text("notes", encoding="utf-8")
                (uploads / "helper.lean").write_text("theorem helper : True := by trivial", encoding="utf-8")
                (uploads / "keep.bin").write_bytes(b"\x00\x01")

                with mock.patch.object(system_config, "user_uploads_dir", str(uploads)):
                    deleted = await _clear_uploaded_files()

                self.assertEqual(deleted, 2)
                self.assertFalse((uploads / "notes.txt").exists())
                self.assertFalse((uploads / "helper.lean").exists())
                self.assertTrue((uploads / "keep.bin").exists())

        asyncio.run(run_case())


class PaperLibraryPathHardeningTests(TestCase):
    def _library_for(self, base_dir: Path) -> PaperLibrary:
        library = PaperLibrary()
        library._base_dir = base_dir
        library._archive_dir = base_dir / "archive"
        library._pruned_dir = base_dir / "pruned"
        base_dir.mkdir(parents=True, exist_ok=True)
        library._archive_dir.mkdir(parents=True, exist_ok=True)
        library._pruned_dir.mkdir(parents=True, exist_ok=True)
        return library

    def test_paper_library_rejects_malicious_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            library = self._library_for(Path(temp_dir) / "papers")

            for paper_id in ("../evil", "a/b", r"a\b", ".", ".."):
                with self.subTest(paper_id=paper_id):
                    with self.assertRaises(ValueError):
                        library.get_paper_path(paper_id)

    def test_prune_paper_keeps_outputs_in_session_pruned_directory(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir = Path(temp_dir) / "session" / "papers"
                library = self._library_for(base_dir)
                metadata = PaperMetadata(paper_id="paper_1", title="Test Paper")
                (base_dir / "paper_paper_1.txt").write_text("paper content", encoding="utf-8")
                (base_dir / "paper_paper_1_metadata.json").write_text(
                    json.dumps(metadata.model_dump(), default=str),
                    encoding="utf-8",
                )

                self.assertTrue(await library.prune_paper("paper_1", reason="duplicate", pruned_by="user"))

                self.assertTrue((base_dir / "pruned" / "pruned_paper_paper_1.txt").exists())
                self.assertTrue((base_dir / "pruned" / "pruned_paper_paper_1_metadata.json").exists())
                self.assertFalse((base_dir / "paper_paper_1.txt").exists())

        asyncio.run(run_case())


class ProofDatabasePathHardeningTests(TestCase):
    def _database_for(self, base_dir: Path) -> ProofDatabase:
        database = ProofDatabase()
        database.set_base_dir(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        return database

    def test_proof_paths_reject_traversal_and_both_separator_styles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = self._database_for(Path(temp_dir) / "proofs")

            for proof_id in ("../evil", r"..\evil", "a/b", r"a\b", ".", ".."):
                with self.subTest(proof_id=proof_id):
                    with self.assertRaises(ValueError):
                        database._get_record_path(proof_id)
                    with self.assertRaises(ValueError):
                        database._get_lean_path(proof_id)

    def test_filename_resolver_rejects_absolute_and_drive_like_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "proofs"
            root.mkdir()
            for filename in ("/tmp/evil", r"C:\temp\evil", "../evil", r"..\evil"):
                with self.subTest(filename=filename):
                    with self.assertRaises(ValueError):
                        resolve_filename_within_root(root, filename)
            self.assertEqual(
                resolve_filename_within_root(root, "proof_001.json"),
                (root / "proof_001.json").resolve(),
            )

    def test_proof_paths_remain_inside_active_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "proofs"
            database = self._database_for(base_dir)

            self.assertEqual(
                database._get_record_path("proof_001"),
                (base_dir / "proof_proof_001.json").resolve(),
            )
            self.assertEqual(
                database._get_lean_path("proof_001"),
                (base_dir / "proof_proof_001_lean.lean").resolve(),
            )
            self.assertEqual(
                database._get_failed_candidates_path("topic_001"),
                (base_dir / "failed" / "topic_001.json").resolve(),
            )

    def test_failed_candidate_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = self._database_for(Path(temp_dir) / "proofs")

            for brainstorm_id in ("../evil", r"..\evil", "a/b", r"a\b", ".", ".."):
                with self.subTest(brainstorm_id=brainstorm_id):
                    with self.assertRaises(ValueError):
                        database._get_failed_candidates_path(brainstorm_id)

    def test_occurrence_write_rejects_malicious_id_without_creating_artifacts(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir = Path(temp_dir) / "proofs"
                database = self._database_for(base_dir)
                record = ProofRecord(
                    proof_id="../outside",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper_1",
                    lean_code="theorem safe : True := by trivial",
                )
                with self.assertRaises(ValueError):
                    await database.add_proof_occurrence(record)
                self.assertFalse((Path(temp_dir) / "proof_outside.json").exists())
                self.assertEqual(list(base_dir.glob("proof_*")), [])

        asyncio.run(run_case())

    def test_occurrence_write_creates_both_valid_artifacts(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir = Path(temp_dir) / "proofs"
                database = self._database_for(base_dir)
                record = ProofRecord(
                    proof_id="proof_custom",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper_1",
                    lean_code="theorem safe : True := by trivial",
                )
                await database.add_proof_occurrence(record)
                self.assertTrue((base_dir / "proof_proof_custom.json").exists())
                self.assertTrue((base_dir / "proof_proof_custom_lean.lean").exists())

        asyncio.run(run_case())


class BrainstormMemoryPathHardeningTests(TestCase):
    def _memory_for(self, base_dir: Path) -> BrainstormMemory:
        memory = BrainstormMemory()
        memory._base_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        return memory

    def test_delete_brainstorm_rejects_topic_id_glob_characters(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir = Path(temp_dir) / "brainstorms"
                memory = self._memory_for(base_dir)

                unrelated_rejection = base_dir / "brainstorm_topic_a_submitter_1_rejections.txt"
                unrelated_rejection.write_text("unrelated", encoding="utf-8")

                self.assertFalse(await memory.delete_brainstorm("topic_[ab]"))

                self.assertTrue(unrelated_rejection.exists())

        asyncio.run(run_case())

    def test_public_brainstorm_routes_return_400_for_invalid_topic_ids(self) -> None:
        async def run_case() -> None:
            with self.assertRaises(HTTPException) as get_context:
                await autonomous_route.get_brainstorm("topic_[ab]")
            self.assertEqual(get_context.exception.status_code, 400)

            with self.assertRaises(HTTPException) as delete_context:
                await autonomous_route.delete_brainstorm("topic_[ab]", confirm=True)
            self.assertEqual(delete_context.exception.status_code, 400)

        asyncio.run(run_case())

    def test_delete_brainstorm_removes_valid_topic_rejection_logs_only(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir = Path(temp_dir) / "brainstorms"
                memory = self._memory_for(base_dir)

                matching_rejection = base_dir / "brainstorm_topic_001_submitter_1_rejections.txt"
                other_rejection = base_dir / "brainstorm_topic_002_submitter_1_rejections.txt"
                malformed_rejection = base_dir / "brainstorm_topic_001_submitter_x_rejections.txt"
                matching_rejection.write_text("matching", encoding="utf-8")
                other_rejection.write_text("other", encoding="utf-8")
                malformed_rejection.write_text("malformed", encoding="utf-8")

                self.assertTrue(await memory.delete_brainstorm("topic_001"))

                self.assertFalse(matching_rejection.exists())
                self.assertTrue(other_rejection.exists())
                self.assertTrue(malformed_rejection.exists())

        asyncio.run(run_case())


class AutonomousRejectionLogsPathHardeningTests(TestCase):
    def _logs_for(self, base_dir: Path) -> AutonomousRejectionLogs:
        logs = AutonomousRejectionLogs()
        logs._brainstorms_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        return logs

    def test_rejection_log_paths_reject_invalid_topic_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs = self._logs_for(Path(temp_dir) / "brainstorms")

            for topic_id in ("../evil", r"evil\path", "topic_[ab]"):
                with self.subTest(topic_id=topic_id):
                    with self.assertRaises(ValueError):
                        logs._get_completion_feedback_path(topic_id)
                    with self.assertRaises(ValueError):
                        logs._get_submitter_rejections_path(topic_id, 1)

    def test_rejection_log_paths_accept_generated_topic_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "brainstorms"
            logs = self._logs_for(base_dir)

            self.assertEqual(
                logs._get_completion_feedback_path("topic_001"),
                base_dir / "completion_feedback_topic_001.txt",
            )
            self.assertEqual(
                logs._get_submitter_rejections_path("topic_001", 3),
                base_dir / "brainstorm_topic_001_submitter_3_rejections.txt",
            )
