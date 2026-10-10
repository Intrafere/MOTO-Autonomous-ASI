from __future__ import annotations

import json

import pytest

from backend.shared import atomic_json
from backend.shared.atomic_json import atomic_write_json


def test_atomic_write_json_writes_sorted_payload(tmp_path):
    path = tmp_path / "nested" / "state.json"

    atomic_write_json(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"b"')
    assert [item.name for item in path.parent.iterdir()] == ["state.json"]


def test_failed_replace_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": 1})

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(atomic_json.os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["state.json"]


def test_unserializable_payload_never_touches_existing_file(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": 1})

    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": object()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["state.json"]
