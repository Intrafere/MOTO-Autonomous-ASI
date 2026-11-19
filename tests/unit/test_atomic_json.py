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


def _locked_then_released(locked_attempts, real_replace):
    calls = []

    def replace(source, destination):
        calls.append(source)
        if len(calls) <= locked_attempts:
            raise PermissionError(5, "Access is denied")
        real_replace(source, destination)

    return replace, calls


def test_windows_sharing_lock_on_replace_is_waited_out(tmp_path, monkeypatch):
    path = tmp_path / "brainstorm_topic_001_metadata.json"
    atomic_write_json(path, {"version": 1})
    replace, calls = _locked_then_released(3, atomic_json.os.replace)
    sleeps = []
    monkeypatch.setattr(atomic_json, "_REPLACE_RETRIES_ON_PERMISSION_ERROR", True)
    monkeypatch.setattr(atomic_json.os, "replace", replace)
    monkeypatch.setattr(atomic_json.time, "sleep", sleeps.append)

    atomic_write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert len(calls) == 4
    assert sleeps == list(atomic_json._REPLACE_RETRY_DELAYS_SECONDS[:3])
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


def test_lasting_windows_lock_raises_and_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": 1})
    replace, calls = _locked_then_released(100, atomic_json.os.replace)
    monkeypatch.setattr(atomic_json, "_REPLACE_RETRIES_ON_PERMISSION_ERROR", True)
    monkeypatch.setattr(atomic_json.os, "replace", replace)
    monkeypatch.setattr(atomic_json.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError):
        atomic_json.atomic_write_text(path, "{}")

    assert len(calls) == len(atomic_json._REPLACE_RETRY_DELAYS_SECONDS) + 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["state.json"]


def test_permission_error_is_not_retried_off_windows(tmp_path, monkeypatch):
    replace, calls = _locked_then_released(1, atomic_json.os.replace)
    monkeypatch.setattr(atomic_json, "_REPLACE_RETRIES_ON_PERMISSION_ERROR", False)
    monkeypatch.setattr(atomic_json.os, "replace", replace)

    with pytest.raises(PermissionError):
        atomic_json.replace_file(tmp_path / "a", tmp_path / "b")

    assert len(calls) == 1
