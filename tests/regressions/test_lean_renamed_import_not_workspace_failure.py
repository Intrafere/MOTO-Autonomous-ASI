"""A proof importing a renamed Mathlib module is a proof error, not a broken workspace.

Lean reports `import Mathlib.Topology.Instances.Real` (split into
`Mathlib.Topology.Instances.Real.Lemmas`) as a missing `.olean`. That once
wiped and re-downloaded the Mathlib workspace, then stopped the whole
Autonomous run with a Lean repair stop.
"""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.shared.config import system_config
from backend.shared.lean4_client import _BAD_IMPORT_HINT, Lean4Client, is_lean_workspace_error_text

RENAMED_MODULE = "Mathlib.Topology.Instances.Real"


def _mathlib_package(workspace: Path) -> Path:
    package = workspace / ".lake" / "packages" / "mathlib"
    lemmas = package / "Mathlib" / "Topology" / "Instances" / "Real" / "Lemmas.lean"
    lemmas.parent.mkdir(parents=True)
    lemmas.write_text("-- current module\n", encoding="utf-8")
    return package


def _missing_olean_output(package: Path) -> str:
    olean = package / ".lake" / "build" / "lib" / "lean" / "Mathlib" / "Topology" / "Instances" / "Real.olean"
    return (
        "MOTOProofCheck_x.lean:1:0: error: object file "
        f"'{olean}' of module {RENAMED_MODULE} does not exist"
    )


def test_missing_olean_without_source_is_a_bad_import(tmp_path):
    output = _missing_olean_output(_mathlib_package(tmp_path))

    assert not Lean4Client._is_workspace_infrastructure_error(output)
    assert _BAD_IMPORT_HINT in Lean4Client._annotate_bad_import_hint(output)


def test_missing_olean_with_existing_source_is_still_a_workspace_failure(tmp_path):
    package = _mathlib_package(tmp_path)
    (package / "Mathlib" / "Topology" / "Instances" / "Real.lean").write_text("", encoding="utf-8")

    assert Lean4Client._is_workspace_infrastructure_error(_missing_olean_output(package))


def test_missing_olean_with_missing_package_is_still_a_workspace_failure(tmp_path):
    package = tmp_path / ".lake" / "packages" / "mathlib"

    assert Lean4Client._is_workspace_infrastructure_error(_missing_olean_output(package))


@pytest.mark.asyncio
async def test_check_proof_returns_proof_feedback_without_repairing(tmp_path, monkeypatch):
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    output = _missing_olean_output(_mathlib_package(tmp_path))
    client = Lean4Client(lean_path=str(tmp_path / "lean.exe"), workspace_dir=str(tmp_path))
    monkeypatch.setattr(client, "ensure_workspace", AsyncMock(return_value=True))
    monkeypatch.setattr(client, "_run_lean_file_once", AsyncMock(return_value=(1, "", output)))
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(client, "_repair_workspace_after_infrastructure_error", repair)

    result = await client.check_proof(
        f"import {RENAMED_MODULE}\n\ntheorem t : (1 : ℝ) = 1 := rfl\n"
    )

    repair.assert_not_awaited()
    assert not result.success
    assert not is_lean_workspace_error_text(result.error_output)
    assert _BAD_IMPORT_HINT in result.error_output
    assert RENAMED_MODULE in result.error_output
    client.require_workspace()
