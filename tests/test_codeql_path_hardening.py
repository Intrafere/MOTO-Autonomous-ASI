import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from backend.aggregator.core.coordinator import _resolve_uploaded_user_file
from backend.aggregator.ingestion.pipeline import IngestionPipeline
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.shared.config import system_config
from backend.shared.models import PaperMetadata


class IngestionPathHardeningTests(unittest.TestCase):
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


class UploadPathResolutionTests(unittest.TestCase):
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


class PaperLibraryPathHardeningTests(unittest.TestCase):
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
