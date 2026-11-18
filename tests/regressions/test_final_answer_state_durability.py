"""Tier 3 state must survive an interrupted save and an unreadable state file.

The state file is rewritten on every tracked model call. It used to be opened
for truncating write, so a kill mid-write left broken JSON, the loader fell
back to defaults, and the next save erased the Tier 3 progress.
"""
import inspect
import json

import pytest

from backend.autonomous.memory import final_answer_memory as module
from backend.autonomous.memory.final_answer_memory import FinalAnswerMemory
from backend.shared import atomic_json


def _memory_in(tmp_path) -> FinalAnswerMemory:
    memory = FinalAnswerMemory()
    memory._session_manager = None
    memory._base_dir = tmp_path
    memory._state_path = tmp_path / "final_answer_state.json"
    memory._volume_path = tmp_path / "volume_organization.json"
    memory._rejections_path = tmp_path / "tier3_rejections.txt"
    memory._final_volume_path = tmp_path / "final_volume.txt"
    return memory


def test_module_has_no_truncating_file_writes():
    source = inspect.getsource(module)
    assert ", 'w'," not in source
    assert ', "w",' not in source


@pytest.mark.asyncio
async def test_interrupted_save_leaves_the_previous_state_file_intact(tmp_path, monkeypatch):
    memory = _memory_in(tmp_path)
    await memory.initialize()
    memory._state.status = "writing"
    await memory._save_state()
    saved = memory._state_path.read_text(encoding="utf-8")
    assert json.loads(saved)["status"] == "writing"

    def fail_replace(source, destination):
        raise OSError("simulated crash before the new file was published")

    monkeypatch.setattr(atomic_json.os, "replace", fail_replace)
    memory._state.status = "complete"
    await memory._save_state()

    assert memory._state_path.read_text(encoding="utf-8") == saved
    assert [path.name for path in tmp_path.iterdir()] == ["final_answer_state.json"]


@pytest.mark.asyncio
async def test_unreadable_state_file_is_preserved_before_defaults_replace_it(tmp_path):
    memory = _memory_in(tmp_path)
    memory._state_path.write_text('{"is_active": true, "status": "writ', encoding="utf-8")

    await memory.initialize()

    assert memory._state.is_active is False
    preserved = list(tmp_path.glob("final_answer_state.json.corrupt-*"))
    assert len(preserved) == 1
    assert preserved[0].read_text(encoding="utf-8") == '{"is_active": true, "status": "writ'
    assert not memory._state_path.exists()
