import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_codeql
from scripts.run_codeql import (
    ACTIONS_PATH_FILTERS,
    Alert,
    export_tree,
    find_codeql,
    load_alerts,
    new_alerts,
    scan_tree,
    snapshot_tree,
)


def _alert(rule="py/x", path="a.py", line=1, message="Problem.", source_line="x = 1", default_suite=True):
    return Alert(
        language="python",
        rule=rule,
        severity="high (7.5)",
        path=path,
        line=line,
        message=message,
        source_line=source_line,
        default_suite=default_suite,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "core.autocrlf", "false")
    (root / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "tracked.py")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_load_alerts_keeps_first_message_line_source_text_and_severity(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "mod.py").write_text("import hashlib\n  digest = hashlib.sha256(key)\n", encoding="utf-8")
    sarif = {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "rules": [
                            {"id": "py/weak", "properties": {"security-severity": "7.5"}},
                            {"id": "py/note", "defaultConfiguration": {"level": "note"}},
                        ]
                    }
                },
                "results": [
                    {
                        "ruleId": "py/weak",
                        "message": {"text": "[Secret](1) is hashed.\n[Secret](2) is hashed."},
                        "locations": [
                            {"physicalLocation": {"artifactLocation": {"uri": "pkg/mod.py"}, "region": {"startLine": 2}}}
                        ],
                    },
                    {
                        "ruleId": "py/note",
                        "message": {"text": "Note."},
                        "locations": [
                            {"physicalLocation": {"artifactLocation": {"uri": "pkg/mod.py"}, "region": {"startLine": 1}}}
                        ],
                    },
                ],
            }
        ]
    }
    sarif_path = tmp_path / "python.sarif"
    sarif_path.write_text(json.dumps(sarif), encoding="utf-8")

    alerts = load_alerts(sarif_path, "python", source, default_rules={"py/weak"})

    assert alerts[0] == Alert(
        language="python",
        rule="py/weak",
        severity="high (7.5)",
        path="pkg/mod.py",
        line=2,
        message="[Secret] is hashed.",
        source_line="digest = hashlib.sha256(key)",
        default_suite=True,
    )
    assert alerts[1].severity == "note"
    assert alerts[1].default_suite is False


def test_moved_lines_and_changed_source_lists_are_not_new() -> None:
    baseline = [_alert(line=10)]
    candidate = [_alert(line=42)]
    assert new_alerts(candidate, baseline) == []


def test_new_alerts_count_duplicates_and_report_real_changes() -> None:
    baseline = [_alert()]
    duplicate = _alert(line=9)
    changed_code = _alert(source_line="x = hash(secret)")
    other_rule = _alert(rule="js/proto", path="b.js")

    introduced = new_alerts([_alert(), duplicate, changed_code, other_rule], baseline)

    assert introduced == sorted([duplicate, changed_code, other_rule], key=lambda a: (a.path, a.line, a.rule))


def test_working_tree_snapshot_includes_untracked_files_without_touching_the_index(repo: Path) -> None:
    clean_tree, clean_description = snapshot_tree(None, repo=repo)
    assert clean_tree == _git(repo, "rev-parse", "HEAD^{tree}")
    assert clean_description.startswith("HEAD")

    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    status_before = _git(repo, "status", "--porcelain")

    tree, description = snapshot_tree(None, repo=repo)

    assert "working tree" in description
    assert set(_git(repo, "ls-tree", "-r", "--name-only", tree).splitlines()) == {"tracked.py", "new_module.py"}
    assert _git(repo, "show", f"{tree}:tracked.py") == "x = 2"
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_export_tree_writes_exactly_the_tree(repo: Path, tmp_path: Path) -> None:
    (repo / "untracked.txt").write_text("not committed\n", encoding="utf-8")
    destination = tmp_path / "export"

    export_tree(_git(repo, "rev-parse", "HEAD^{tree}"), destination, repo=repo)

    assert sorted(path.name for path in destination.iterdir()) == ["tracked.py"]
    assert (destination / "tracked.py").read_text(encoding="utf-8") == "x = 1\n"


def test_find_codeql_prefers_explicit_then_env_then_installed(tmp_path: Path, monkeypatch) -> None:
    for name in ("LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(tmp_path / "cache"))
    monkeypatch.delenv("MOTO_CODEQL", raising=False)
    monkeypatch.delenv("RUNNER_TOOL_CACHE", raising=False)
    monkeypatch.setattr(run_codeql.shutil, "which", lambda _name: None)
    exe = run_codeql._codeql_executable_name()
    pinned = run_codeql.tools_root() / f"v{run_codeql.CODEQL_BUNDLE_VERSION}" / "codeql" / exe
    assert find_codeql() is None

    pinned.parent.mkdir(parents=True)
    pinned.write_text("", encoding="utf-8")
    assert find_codeql() == pinned

    chosen = tmp_path / "custom" / exe
    chosen.parent.mkdir()
    chosen.write_text("", encoding="utf-8")
    monkeypatch.setenv("MOTO_CODEQL", str(chosen))
    assert find_codeql() == chosen
    assert find_codeql(str(pinned)) == pinned
    assert find_codeql(str(tmp_path / "missing")) is None


@pytest.mark.parametrize("user_filters", ["", "exclude:vendor/**"])
def test_actions_autobuild_points_the_javascript_extractor_at_workflow_files(
    tmp_path: Path, monkeypatch, user_filters: str
) -> None:
    js_root = tmp_path / "javascript"
    monkeypatch.setenv("CODEQL_DIST", str(tmp_path / "dist"))
    monkeypatch.setenv("LGTM_INDEX_FILTERS", user_filters)
    monkeypatch.setenv("CODEQL_EXTRACTOR_JAVASCRIPT_ROOT", "placeholder")
    for suffix in run_codeql._ACTIONS_EXTRACTOR_DIRS:
        monkeypatch.setenv(f"CODEQL_EXTRACTOR_ACTIONS_{suffix}", f"actions-{suffix}")
        monkeypatch.setenv(f"CODEQL_EXTRACTOR_JAVASCRIPT_{suffix}", "placeholder")

    class Resolved:
        stdout = f"WARNING: noise\n{js_root}\n"

    calls = []
    monkeypatch.setattr(run_codeql.subprocess, "run", lambda *args, **kwargs: Resolved())
    monkeypatch.setattr(run_codeql.subprocess, "call", lambda command: calls.append(command) or 0)

    assert run_codeql.run_actions_autobuild() == 0

    expected_filters = "\n".join(ACTIONS_PATH_FILTERS) + (f"\n{user_filters}" if user_filters else "")
    assert run_codeql.os.environ["LGTM_INDEX_FILTERS"] == expected_filters
    assert run_codeql.os.environ["CODEQL_EXTRACTOR_JAVASCRIPT_ROOT"] == str(js_root)
    for suffix in run_codeql._ACTIONS_EXTRACTOR_DIRS:
        assert run_codeql.os.environ[f"CODEQL_EXTRACTOR_JAVASCRIPT_{suffix}"] == f"actions-{suffix}"
    assert calls and "autobuild" in calls[0][-1]


def test_scan_tree_reuses_results_for_an_identical_tree(tmp_path: Path) -> None:
    cache = tmp_path / "results" / f"{'a' * 40}-codeql-9.9.9-v{run_codeql.RESULTS_FORMAT}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([_alert().__dict__]), encoding="utf-8")

    alerts, cached = scan_tree(
        "a" * 40,
        codeql=tmp_path / "missing-codeql",
        version="9.9.9",
        work_dir=tmp_path,
        label="candidate",
        keep=False,
    )

    assert cached is True
    assert alerts == [_alert()]


@pytest.mark.parametrize("introduce", [False, True])
def test_main_fails_only_when_the_candidate_adds_alerts(monkeypatch, capsys, introduce: bool) -> None:
    existing = _alert()
    added = _alert(rule="js/proto", path="b.js", default_suite=False)
    monkeypatch.setattr(run_codeql, "find_codeql", lambda explicit=None: Path("codeql"))
    monkeypatch.setattr(run_codeql, "codeql_version", lambda codeql: "9.9.9")
    monkeypatch.setattr(run_codeql, "snapshot_tree", lambda rev, repo=None: (f"{rev or 'work'}-tree", rev or "working tree"))
    monkeypatch.setattr(
        run_codeql,
        "scan_tree",
        lambda tree, label, **kwargs: ([existing, added] if introduce and label == "candidate" else [existing], False),
    )

    exit_code = run_codeql.main(["--no-fetch"])

    output = capsys.readouterr().out
    if introduce:
        assert exit_code == 1
        assert "FAIL: 1 new CodeQL alert" in output
        assert "js/proto  b.js:1  [security-extended suite only]" in output
    else:
        assert exit_code == 0
        assert "PASS" in output


def test_main_reports_a_missing_codeql_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_codeql, "find_codeql", lambda explicit=None: None)

    assert run_codeql.main(["--no-fetch"]) == 2
    assert "--install" in capsys.readouterr().err
