from __future__ import annotations

import pytest

from tests.run_all_tests import (
    ROOT_DIR,
    TESTS_DIR,
    TestFileResult as FileRunResult,
    discover_test_files,
    parse_pytest_args,
    resolve_requested_files,
)


def test_parse_pytest_args_preserves_quoted_expression() -> None:
    assert parse_pytest_args('-k "workflow and not slow" --maxfail 1') == [
        "-k",
        "workflow and not slow",
        "--maxfail",
        "1",
    ]


def test_parse_pytest_args_accepts_empty_input() -> None:
    assert parse_pytest_args("") == []


def test_parse_pytest_args_preserves_windows_backslashes() -> None:
    assert parse_pytest_args(r'--basetemp "C:\temp\pytest run"') == [
        "--basetemp",
        r"C:\temp\pytest run",
    ]


def test_pytest_no_match_exit_is_successful_skip() -> None:
    result = FileRunResult(
        path=TESTS_DIR / "test_example.py",
        returncode=5,
        elapsed_seconds=0.1,
        stdout="",
        stderr="",
        allow_no_tests=True,
    )

    assert result.skipped
    assert result.successful
    assert not result.passed


def test_requested_file_must_be_a_pytest_file_under_tests() -> None:
    with pytest.raises(SystemExit, match="Not a runnable pytest file"):
        resolve_requested_files([str(ROOT_DIR / "package.json")])


def test_discovery_includes_nested_test_packages() -> None:
    discovered = {path.relative_to(ROOT_DIR).as_posix() for path in discover_test_files()}

    assert "tests/unit/test_build_info.py" in discovered
    assert "tests/integration/test_allowed_outputs.py" in discovered
    assert "tests/regressions/test_context_overflow_metadata.py" in discovered
    assert "tests/workflow_scenarios/test_workflow_scenarios.py" in discovered


def test_requested_nested_test_file_is_resolved() -> None:
    relative_path = "tests/integration/test_allowed_outputs.py"

    assert resolve_requested_files([relative_path]) == [(ROOT_DIR / relative_path).resolve()]
