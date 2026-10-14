from pathlib import Path

from backend.shared.lean_workspace_paths import (
    MAX_SAFE_WORKSPACE_PATH,
    data_root_workspace_key,
    is_moto_lean_workspace,
    lean_workspace_dir_for,
)


def _long_root(tmp_path: Path) -> Path:
    return tmp_path / ("deep" * 40) / "backend" / "data"


def _short_local_app_data(tmp_path: Path) -> Path:
    # Only computed, never created: the resolver never touches the filesystem.
    return Path(tmp_path.anchor) / "la"


def test_short_data_root_keeps_workspace_inside_it(tmp_path):
    data_root = Path(tmp_path.anchor) / "MOTO" / "backend" / "data"
    assert lean_workspace_dir_for(
        data_root, is_windows=True, local_app_data=str(_short_local_app_data(tmp_path)),
    ) == data_root / "lean4_workspace"


def test_long_windows_data_root_moves_workspace_to_short_per_user_folder(tmp_path):
    data_root = _long_root(tmp_path)
    local_app_data = _short_local_app_data(tmp_path)

    workspace = lean_workspace_dir_for(
        data_root, is_windows=True, local_app_data=str(local_app_data),
    )

    assert workspace == local_app_data / "MOTO" / "lean" / data_root_workspace_key(data_root)
    assert len(str(workspace)) <= MAX_SAFE_WORKSPACE_PATH


def test_workspace_key_is_stable_and_separates_installs(tmp_path):
    first = _long_root(tmp_path)
    second = tmp_path / ("other" * 40) / "backend" / "data"

    assert data_root_workspace_key(first) == data_root_workspace_key(Path(str(first)))
    assert data_root_workspace_key(first) != data_root_workspace_key(second)
    assert len(data_root_workspace_key(first)) == 12


def test_relocation_falls_back_in_root_when_no_short_folder_is_available(tmp_path):
    data_root = _long_root(tmp_path)
    in_root = data_root / "lean4_workspace"

    assert lean_workspace_dir_for(data_root, is_windows=True, local_app_data="") == in_root
    assert lean_workspace_dir_for(
        data_root, is_windows=True, local_app_data=str(tmp_path / ("x" * 120)),
    ) == in_root


def test_other_platforms_never_relocate(tmp_path):
    data_root = _long_root(tmp_path)
    assert lean_workspace_dir_for(
        data_root, is_windows=False, local_app_data=str(tmp_path / "appdata"),
    ) == data_root / "lean4_workspace"


def test_only_moto_workspaces_are_recognised(tmp_path):
    workspace = tmp_path / "lean4_workspace"
    workspace.mkdir()
    assert not is_moto_lean_workspace(workspace)
    (workspace / "MOTOProofWorkspace.lean").write_text("import Mathlib\n", encoding="utf-8")
    assert is_moto_lean_workspace(workspace)
