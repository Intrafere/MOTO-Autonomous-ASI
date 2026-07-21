from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend.shared.runtime_root_lock import RuntimeRootInUseError, RuntimeRootLease


def test_runtime_root_lease_rejects_same_process_owner(tmp_path: Path) -> None:
    first = RuntimeRootLease(tmp_path).acquire()
    try:
        with pytest.raises(RuntimeRootInUseError):
            RuntimeRootLease(tmp_path).acquire()
    finally:
        first.release()
        first.release()

    replacement = RuntimeRootLease(tmp_path).acquire()
    replacement.release()


def test_runtime_root_lease_allows_distinct_roots(tmp_path: Path) -> None:
    first = RuntimeRootLease(tmp_path / "one").acquire()
    second = RuntimeRootLease(tmp_path / "two").acquire()
    second.release()
    first.release()


@pytest.mark.skipif(os.name != "nt", reason="Windows lock regression")
def test_runtime_root_lease_rejects_competing_process_then_recovers(tmp_path: Path) -> None:
    script = """
import sys
from backend.shared.runtime_root_lock import RuntimeRootLease
lease = RuntimeRootLease(sys.argv[1]).acquire()
print("LOCKED", flush=True)
sys.stdin.readline()
lease.release()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeRootInUseError):
            RuntimeRootLease(tmp_path).acquire()
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        child.wait(timeout=10)

    recovered = RuntimeRootLease(tmp_path).acquire()
    recovered.release()
