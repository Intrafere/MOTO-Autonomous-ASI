import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.aggregator.core.chroma_cache import (
    complete_chroma_cache_rebuild,
    maintain_chroma_cache_directory,
    quarantine_chroma_cache,
    recover_interrupted_chroma_rebuild,
)


def _create_chroma_sqlite(path: Path, referenced_ids: list[str]) -> None:
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("create table collections (id text, name text)")
        cur.execute("create table segments (id text, type text, scope text, collection text)")
        for index, referenced_id in enumerate(referenced_ids):
            cur.execute(
                "insert into segments values (?, ?, ?, ?)",
                (referenced_id, "urn:chroma:segment/vector/hnsw-local-persisted", "VECTOR", f"collection-{index}"),
            )
        con.commit()
    finally:
        con.close()


class ChromaCacheCleanupTests(unittest.TestCase):
    def test_removes_only_chroma_cache_when_orphan_uuid_buildup_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            chroma_dir = data_root / "chroma_db"
            chroma_dir.mkdir()

            durable_file = data_root / "rag_shared_training.txt"
            durable_file.write_text("keep me", encoding="utf-8")
            durable_session = data_root / "auto_sessions" / "session_1"
            durable_session.mkdir(parents=True)
            (durable_session / "paper.txt").write_text("keep session", encoding="utf-8")

            _create_chroma_sqlite(chroma_dir / "chroma.sqlite3", referenced_ids=[])
            for i in range(70):
                orphan_dir = chroma_dir / f"00000000-0000-0000-0000-{i:012x}"
                orphan_dir.mkdir()
                (orphan_dir / "index.bin").write_text("cache", encoding="utf-8")
            (chroma_dir / "not-a-uuid.txt").write_text("cache", encoding="utf-8")

            result = maintain_chroma_cache_directory(chroma_dir, data_root)

            self.assertTrue(result.reset_performed)
            self.assertEqual(result.unreferenced_uuid_dir_count, 70)
            self.assertTrue(chroma_dir.exists())
            self.assertEqual(list(chroma_dir.iterdir()), [])
            self.assertEqual(durable_file.read_text(encoding="utf-8"), "keep me")
            self.assertEqual((durable_session / "paper.txt").read_text(encoding="utf-8"), "keep session")

    def test_skips_cleanup_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            chroma_dir = data_root / "chroma_db"
            chroma_dir.mkdir()
            _create_chroma_sqlite(chroma_dir / "chroma.sqlite3", referenced_ids=[])
            orphan_dir = chroma_dir / "00000000-0000-0000-0000-000000000001"
            orphan_dir.mkdir()

            result = maintain_chroma_cache_directory(chroma_dir, data_root)

            self.assertFalse(result.reset_performed)
            self.assertEqual(result.reason, "below_threshold")
            self.assertTrue(orphan_dir.exists())

    def test_rejects_chroma_path_outside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            outside = root / "outside_chroma"
            data_root.mkdir()
            outside.mkdir()

            with self.assertRaises(ValueError):
                maintain_chroma_cache_directory(outside, data_root)

    def test_skips_cleanup_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            chroma_dir = data_root / "chroma_db"
            chroma_dir.mkdir()
            for i in range(70):
                orphan_dir = chroma_dir / f"00000000-0000-0000-0000-{i:012x}"
                orphan_dir.mkdir()

            result = maintain_chroma_cache_directory(chroma_dir, data_root)

            self.assertFalse(result.reset_performed)
            self.assertEqual(result.reason, "missing_sqlite_metadata")
            self.assertEqual(len([p for p in chroma_dir.iterdir() if p.is_dir()]), 70)

    def test_rejects_data_root_as_chroma_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)

            with self.assertRaises(ValueError):
                maintain_chroma_cache_directory(data_root, data_root)

    def test_quarantine_rebuild_preserves_durable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            chroma_dir = data_root / "chroma_db"
            chroma_dir.mkdir()
            (chroma_dir / "chroma.sqlite3").write_text("cache", encoding="utf-8")
            durable = data_root / "rag_shared_training.txt"
            durable.write_text("durable", encoding="utf-8")

            fresh, quarantine = quarantine_chroma_cache(chroma_dir, data_root)
            self.assertEqual(list(fresh.iterdir()), [])
            self.assertIsNotNone(quarantine)
            self.assertTrue((quarantine / "chroma.sqlite3").exists())
            complete_chroma_cache_rebuild(fresh, data_root, quarantine)

            self.assertFalse(quarantine.exists())
            self.assertEqual(durable.read_text(encoding="utf-8"), "durable")

    def test_interrupted_rebuild_restores_last_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            chroma_dir = data_root / "chroma_db"
            chroma_dir.mkdir()
            (chroma_dir / "old.cache").write_text("old", encoding="utf-8")

            fresh, quarantine = quarantine_chroma_cache(chroma_dir, data_root)
            fresh.rmdir()
            recover_interrupted_chroma_rebuild(chroma_dir, data_root)

            self.assertTrue((chroma_dir / "old.cache").exists())
            self.assertFalse(quarantine.exists())


if __name__ == "__main__":
    unittest.main()
