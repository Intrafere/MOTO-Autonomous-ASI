"""Proof-appendix saves must never drop accepted submissions.

Autonomous child phases hand the shared-training singleton back pointed at the
manual database with its in-memory submissions cleared. An appendix-only save
from that state used to rewrite the file without its submissions.
"""
import pytest

from backend.aggregator.memory import shared_training
from backend.aggregator.memory.shared_training import SharedTrainingMemory


async def _database_with_submissions(path, count: int = 3) -> SharedTrainingMemory:
    memory = SharedTrainingMemory()
    memory.file_path = path
    await memory.initialize()
    for index in range(count):
        await memory.add_accepted_submission(f"Accepted submission {index + 1}: " + "x" * 120)
    return memory


def _handed_back_unloaded(path) -> SharedTrainingMemory:
    """A memory pointed at ``path`` whose submissions were never loaded."""
    memory = SharedTrainingMemory()
    memory.file_path = path
    return memory


_PROOF = {
    "theorem_statement": "theorem sample : True",
    "proof_id": "proof_001",
    "novel": True,
    "lean_code": "theorem sample : True := trivial",
}


@pytest.mark.asyncio
async def test_clear_proof_appendix_keeps_submissions_when_memory_is_unloaded(tmp_path):
    path = tmp_path / "rag_shared_training.txt"
    loaded = await _database_with_submissions(path)
    assert await loaded.append_proofs_section(_PROOF)
    assert "Proof ID: proof_001" in path.read_text(encoding="utf-8")

    stale = _handed_back_unloaded(path)
    await stale.clear_proof_appendix()

    content = path.read_text(encoding="utf-8")
    assert content.count("SUBMISSION #") == 3
    assert "Proof ID: proof_001" not in content
    assert len(stale.insights) == 3
    assert stale.submission_count == 3


@pytest.mark.asyncio
async def test_append_proofs_section_keeps_submissions_when_memory_is_unloaded(tmp_path):
    path = tmp_path / "rag_shared_training.txt"
    await _database_with_submissions(path)

    stale = _handed_back_unloaded(path)
    assert await stale.append_proofs_section(_PROOF)

    content = path.read_text(encoding="utf-8")
    assert content.count("SUBMISSION #") == 3
    assert "Proof ID: proof_001" in content


@pytest.mark.asyncio
async def test_manual_clear_helper_keeps_submissions_after_singleton_hand_back(tmp_path, monkeypatch):
    path = tmp_path / "rag_shared_training.txt"
    await _database_with_submissions(path)
    monkeypatch.setattr(shared_training.system_config, "shared_training_file", str(path))
    monkeypatch.setattr(shared_training, "shared_training_memory", _handed_back_unloaded(path))

    await shared_training.clear_manual_shared_training_proof_appendix()

    assert path.read_text(encoding="utf-8").count("SUBMISSION #") == 3


@pytest.mark.asyncio
async def test_manual_append_helper_keeps_submissions_after_singleton_hand_back(tmp_path, monkeypatch):
    path = tmp_path / "rag_shared_training.txt"
    await _database_with_submissions(path)
    monkeypatch.setattr(shared_training.system_config, "shared_training_file", str(path))
    monkeypatch.setattr(shared_training, "shared_training_memory", _handed_back_unloaded(path))

    assert await shared_training.append_proof_to_manual_shared_training(_PROOF)

    content = path.read_text(encoding="utf-8")
    assert content.count("SUBMISSION #") == 3
    assert "Proof ID: proof_001" in content


@pytest.mark.asyncio
async def test_emptied_database_stays_empty_after_appendix_clear(tmp_path):
    """An intentionally empty database is not refilled by the unloaded-memory guard."""
    path = tmp_path / "rag_shared_training.txt"
    memory = await _database_with_submissions(path, count=1)
    assert await memory.remove_submission(1, trigger_rechunk=False)

    await memory.clear_proof_appendix()

    assert path.read_text(encoding="utf-8").strip() == ""
    assert memory.insights == []
