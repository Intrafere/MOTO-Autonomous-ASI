"""Crash-safe persistence regressions: atomic text writes, log fd ownership, and
corrupt-metadata preservation."""
from __future__ import annotations

import asyncio
import json
import os
from importlib import import_module
from pathlib import Path

import pytest

from backend.shared import api_log_persistence, atomic_json
from backend.shared.atomic_json import atomic_write_text


def test_atomic_write_text_replaces_file_without_leaving_temp(tmp_path):
    path = tmp_path / "nested" / "compiler_paper.txt"

    atomic_write_text(path, "first")
    atomic_write_text(path, "second draft")

    assert path.read_text(encoding="utf-8") == "second draft"
    assert [item.name for item in path.parent.iterdir()] == ["compiler_paper.txt"]


def test_atomic_write_text_keeps_previous_file_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "rag_shared_training.txt"
    atomic_write_text(path, "accepted submission #1")

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(atomic_json.os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_text(path, "")

    assert path.read_text(encoding="utf-8") == "accepted submission #1"
    assert [item.name for item in tmp_path.iterdir()] == ["rag_shared_training.txt"]


def test_atomic_log_write_never_closes_descriptor_twice(tmp_path, monkeypatch):
    path = tmp_path / "api_calls.jsonl"
    closed: list[int] = []
    real_close = os.close

    def recording_close(fd):
        closed.append(fd)
        real_close(fd)

    def fail_replace(self, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(api_log_persistence.os, "close", recording_close)
    monkeypatch.setattr(api_log_persistence.Path, "replace", fail_replace)

    with pytest.raises(OSError):
        api_log_persistence.atomic_write_log_lines(path, ["{}\n"])

    # fdopen closed the descriptor on its own; the failure path must not close
    # that number again (it may already belong to another thread's file).
    assert len(closed) == 1
    assert not list(tmp_path.glob(".api_calls.jsonl.*.tmp"))


@pytest.mark.asyncio
async def test_unreadable_research_metadata_is_preserved_not_overwritten(tmp_path, monkeypatch):
    module = import_module("backend.autonomous.memory.research_metadata")

    registry = module.ResearchMetadata()
    registry._metadata_path = tmp_path / "research_metadata.json"
    registry._stats_path = tmp_path / "research_stats.json"
    registry._workflow_state_path = tmp_path / "workflow_state.json"
    registry._metadata_path.write_text('{"brainstorms": [{"topic_id": "topic_1"', encoding="utf-8")

    await registry._load_metadata()

    preserved = sorted(item.name for item in tmp_path.iterdir())
    assert "research_metadata.json" in preserved
    corrupt_copies = [name for name in preserved if name.startswith("research_metadata.json.corrupt-")]
    assert len(corrupt_copies) == 1
    assert (tmp_path / corrupt_copies[0]).read_text(encoding="utf-8").startswith('{"brainstorms"')
    assert json.loads(registry._metadata_path.read_text(encoding="utf-8"))["brainstorms"] == []
    assert registry._data["next_topic_id"] == 1


@pytest.mark.asyncio
async def test_metadata_save_is_atomic(tmp_path):
    module = import_module("backend.autonomous.memory.research_metadata")

    registry = module.ResearchMetadata()
    registry._metadata_path = tmp_path / "research_metadata.json"
    registry._data = {"brainstorms": [], "papers": [], "next_topic_id": 3}

    await registry._save_metadata()

    assert json.loads(registry._metadata_path.read_text(encoding="utf-8"))["next_topic_id"] == 3
    assert [item.name for item in tmp_path.iterdir()] == ["research_metadata.json"]
