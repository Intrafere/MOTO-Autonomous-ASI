"""Retroactive brainstorm edits must keep the proof appendix after every submission.

The appendix used to be parsed as part of the last submission, so a retroactive
addition was written after it and every proof-stripped read hid the addition,
and editing or deleting the last submission dropped the appendix from the file.
"""
import pytest

from backend.autonomous.memory.brainstorm_memory import (
    BrainstormMemory,
    split_brainstorm_proof_appendix,
)

SEPARATOR = "=" * 80
APPENDIX_HEADER = "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ==="
APPENDIX = (
    f"{APPENDIX_HEADER}\n\n"
    "Proof 1: theorem sample : True\n"
    "Status: Verified (Novel)\n"
    "Proof ID: proof_001\n"
    "Lean 4 Code:\n"
    "theorem sample : True := trivial\n"
    "---"
)


def _submission(number: int, content: str) -> str:
    return f"{SEPARATOR}\nSUBMISSION #{number} | Accepted: 2026-01-0{number}T00:00:00\n{SEPARATOR}\n\n{content}\n"


def _memory_with_database(tmp_path, content: str) -> tuple[BrainstormMemory, object]:
    memory = BrainstormMemory()
    memory._base_dir = tmp_path
    db_path = memory._get_database_path("topic_001")
    db_path.write_text(content, encoding="utf-8")
    return memory, db_path


def _database_with_appendix() -> str:
    return "\n\n".join([_submission(1, "first idea"), _submission(2, "second idea")]) + f"\n\n{APPENDIX}\n"


def test_split_returns_whole_content_when_there_is_no_appendix():
    content = _submission(1, "first idea")
    assert split_brainstorm_proof_appendix(content) == (content, "")


@pytest.mark.asyncio
async def test_retroactive_add_lands_before_the_appendix_and_stays_model_visible(tmp_path):
    memory, db_path = _memory_with_database(tmp_path, _database_with_appendix())

    assert await memory.add_submission_retroactive("topic_001", "late insight") == 3

    content = db_path.read_text(encoding="utf-8")
    assert content.index("late insight") < content.index(APPENDIX_HEADER)
    assert content.rstrip().endswith(APPENDIX)
    model_view = await memory.get_database_content("topic_001", strip_proofs=True)
    assert "late insight" in model_view
    assert "Proof ID: proof_001" not in model_view


@pytest.mark.asyncio
async def test_editing_the_last_submission_keeps_the_appendix(tmp_path):
    memory, db_path = _memory_with_database(tmp_path, _database_with_appendix())

    assert await memory.edit_submission("topic_001", 2, "second idea, corrected")

    content = db_path.read_text(encoding="utf-8")
    assert "second idea, corrected" in content
    assert content.rstrip().endswith(APPENDIX)
    assert content.count(APPENDIX_HEADER) == 1


@pytest.mark.asyncio
async def test_removing_the_last_submission_keeps_the_appendix(tmp_path):
    memory, db_path = _memory_with_database(tmp_path, _database_with_appendix())

    assert await memory.remove_submission("topic_001", 2)

    content = db_path.read_text(encoding="utf-8")
    assert "second idea" not in content
    assert "first idea" in content
    assert content.rstrip().endswith(APPENDIX)


@pytest.mark.asyncio
async def test_submission_written_after_the_appendix_by_an_older_build_is_recovered(tmp_path):
    damaged = _database_with_appendix() + "\n\n" + _submission(3, "addition hidden behind the proofs")
    memory, db_path = _memory_with_database(tmp_path, damaged)

    model_view = await memory.get_database_content("topic_001", strip_proofs=True)
    assert "addition hidden behind the proofs" in model_view
    assert "Proof ID: proof_001" not in model_view

    # The next retroactive write puts the file back in order.
    assert await memory.add_submission_retroactive("topic_001", "newest insight") == 4
    content = db_path.read_text(encoding="utf-8")
    assert content.index("addition hidden behind the proofs") < content.index(APPENDIX_HEADER)
    assert content.index("newest insight") < content.index(APPENDIX_HEADER)
    assert content.rstrip().endswith(APPENDIX)
