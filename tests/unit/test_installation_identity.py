import json
from pathlib import Path

import pytest

from backend.shared import installation_identity as identity


@pytest.fixture
def identity_store(monkeypatch):
    stored = {"value": None}
    monkeypatch.setattr(
        identity,
        "load_moto_installation_identity_state",
        lambda: stored["value"],
    )
    monkeypatch.setattr(
        identity,
        "store_moto_installation_identity_state",
        lambda value: stored.__setitem__("value", value),
    )
    monkeypatch.setattr(
        identity,
        "clear_moto_installation_identity_state",
        lambda: stored.__setitem__("value", None),
    )
    monkeypatch.setattr(identity.system_config, "generic_mode", False)
    return stored


def test_empty_roots_receive_distinct_bindings_for_one_installation(
    tmp_path: Path,
    identity_store,
) -> None:
    first = identity.ensure_installation_binding(tmp_path / "first")
    second = identity.ensure_installation_binding(tmp_path / "second")

    assert first.installation_id == second.installation_id
    assert first.root_binding_id != second.root_binding_id
    state = json.loads(identity_store["value"])
    assert set(state["root_bindings"]) == {
        first.root_binding_id,
        second.root_binding_id,
    }


def test_copied_or_moved_root_fails_until_relocation_is_approved(
    tmp_path: Path,
    identity_store,
) -> None:
    original = tmp_path / "original"
    binding = identity.ensure_installation_binding(original)
    moved = tmp_path / "moved"
    original.rename(moved)

    with pytest.raises(identity.InstallationBindingMismatchError):
        identity.ensure_installation_binding(moved)

    approved = identity.approve_runtime_root_relocation(moved)
    assert approved == binding
    assert identity.ensure_installation_binding(moved) == binding


def test_populated_unmarked_root_requires_explicit_adoption(
    tmp_path: Path,
    identity_store,
) -> None:
    identity.ensure_installation_binding(tmp_path / "first")
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "proofs_index.json").write_text("{}", encoding="utf-8")

    with pytest.raises(identity.InstallationBindingMismatchError):
        identity.ensure_installation_binding(copied)

    adopted = identity.adopt_unmarked_runtime_root(copied)
    assert identity.ensure_installation_binding(copied) == adopted


def test_reset_changes_installation_identity_without_deleting_root_data(
    tmp_path: Path,
    identity_store,
) -> None:
    root = tmp_path / "data"
    first = identity.ensure_installation_binding(root)
    proof_file = root / "proof.json"
    proof_file.write_text("preserved", encoding="utf-8")

    second = identity.reset_installation_identity(root)

    assert first.installation_id != second.installation_id
    assert proof_file.read_text(encoding="utf-8") == "preserved"


def test_generic_mode_has_no_installation_identity_authority(
    tmp_path: Path,
    identity_store,
    monkeypatch,
) -> None:
    monkeypatch.setattr(identity.system_config, "generic_mode", True)
    with pytest.raises(identity.InstallationIdentityError):
        identity.ensure_installation_binding(tmp_path / "generic")
