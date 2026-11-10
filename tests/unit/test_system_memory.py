import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.shared import system_memory
from backend.shared.system_memory import (
    MemorySnapshot,
    describe_memory_snapshot,
    system_memory_snapshot,
)

_GIB = 1024 ** 3


class SystemMemoryTests(unittest.TestCase):
    def test_commit_snapshot_names_the_windows_commit_limit(self) -> None:
        text = describe_memory_snapshot(
            MemorySnapshot(available_bytes=int(2.8 * _GIB), limit_bytes=int(127.6 * _GIB), kind="commit")
        )
        self.assertIn("2.8 GB of its 127.6 GB memory commit limit free", text)
        self.assertIn("Task Manager", text)

    def test_unknown_memory_adds_no_sentence(self) -> None:
        self.assertEqual(describe_memory_snapshot(None), "")

    def test_linux_meminfo_reports_available_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meminfo = Path(tmpdir) / "meminfo"
            meminfo.write_text("MemTotal:       16384000 kB\nMemAvailable:    4096000 kB\n", encoding="utf-8")
            with mock.patch.object(system_memory, "_is_windows", return_value=False):
                snapshot = system_memory_snapshot(meminfo_path=meminfo)

        self.assertEqual(snapshot, MemorySnapshot(4096000 * 1024, 16384000 * 1024, "physical"))
        self.assertIn("available", describe_memory_snapshot(snapshot))

    def test_unreadable_platform_data_returns_none(self) -> None:
        missing = Path(tempfile.gettempdir()) / "missing-meminfo"
        with mock.patch.object(system_memory, "_is_windows", return_value=False):
            self.assertIsNone(system_memory_snapshot(meminfo_path=missing))


if __name__ == "__main__":
    unittest.main()
