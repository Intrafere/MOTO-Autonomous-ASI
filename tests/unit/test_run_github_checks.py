import json
from pathlib import Path
import re
import subprocess
import sys

from scripts.run_github_checks import (
    PYTEST_ONLY_BOOTSTRAP,
    RELEASE_FOCUSED_TESTS,
    REPO_ROOT,
    SYNTHETICLIB_EVIDENCE,
    Step,
    build_steps,
    format_summary,
    run_steps,
)


CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "syntheticlib-compatibility-release.yml"


def _commands(steps):
    return {" ".join(step.command) for step in steps}


def test_every_ci_command_has_a_local_step() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    local = _commands(build_steps(python="python", npm="npm"))
    for ci_command in (
        "python -m pip check",
        "python -m tests.workflow_cross_field.artifacts --check",
        "python -m scripts.check_syntheticlib_trade_contract --check",
        "python -m pytest",
        "python -m scripts.run_npm_audit",
        "npm run test",
        "npm run build",
        "npm run test:browser",
        "python -m scripts.run_codeql --no-fetch --install-if-missing",
    ):
        assert ci_command in workflow, f"ci.yml no longer runs {ci_command!r}; update run_github_checks"
    assert "python -m scripts.run_codeql" in local
    assert {
        "python -m pip check",
        "python -m tests.workflow_cross_field.artifacts --check",
        "python -m scripts.check_syntheticlib_trade_contract --check",
        "python -m pytest",
        "python -m scripts.run_npm_audit",
        "npm run test",
        "npm run build",
        "npm run test:browser",
    } <= local


def test_syntheticlib_evidence_list_matches_ci() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    block = workflow.split("git ls-files --error-unmatch", 1)[1].split("python -m", 1)[0]
    assert tuple(re.findall(r"syntheticlib-trade/compatibility/\S+?(?=\s|\\|$)", block)) == SYNTHETICLIB_EVIDENCE


def test_release_focused_tests_match_the_release_workflow() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    block = workflow.split("name: Run focused compatibility tests", 1)[1].split("- name:", 1)[0]
    assert tuple(re.findall(r"tests/\S+\.py", block)) == RELEASE_FOCUSED_TESTS
    assert "run: python -m pip install pytest\n" in workflow, (
        "the release workflow no longer installs only pytest; update PYTEST_ONLY_BOOTSTRAP"
    )
    local = {step.command for step in build_steps(python="python", npm="npm")}
    assert ("python", "-c", PYTEST_ONLY_BOOTSTRAP, *RELEASE_FOCUSED_TESTS) in local


def test_no_workflow_first_runs_after_a_merge_to_main() -> None:
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        triggers = workflow.read_text(encoding="utf-8").split("\non:", 1)[1].split("\njobs:", 1)[0]
        assert "\n  pull_request:" in triggers, f"{workflow.name} does not run on pull requests"
        assert '"dev/**"' in triggers, f"{workflow.name} does not run on dev branch pushes"


def test_pytest_only_bootstrap_hides_installed_packages_but_not_pytest() -> None:
    probe = PYTEST_ONLY_BOOTSTRAP.replace(
        "raise SystemExit(pytest.main(sys.argv[1:]))",
        "import importlib, json, pluggy, scripts.run_github_checks\n"
        "hidden = []\n"
        "for name in ('pydantic', 'httpx'):\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError:\n"
        "        hidden.append(name)\n"
        "print(json.dumps(hidden))\n",
    )
    assert probe != PYTEST_ONLY_BOOTSTRAP
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == ["pydantic", "httpx"]


def test_run_steps_continues_after_a_failure_and_honors_skip(tmp_path: Path) -> None:
    steps = (
        Step("fails", "Fails", (sys.executable, "-c", "raise SystemExit(3)")),
        Step("skipped", "Skipped", (sys.executable, "-c", "raise SystemExit(4)")),
        Step("passes", "Passes", (sys.executable, "-c", "pass")),
    )

    outcomes = run_steps(steps, skip=["skipped"], repo=tmp_path)

    assert [status for _, status, _ in outcomes] == ["FAILED (exit 3)", "skipped", "passed"]
    summary = format_summary(outcomes)
    assert "NOT READY TO PUSH: 1 check(s) failed: fails" in summary


def test_summary_names_skipped_checks_instead_of_claiming_ready() -> None:
    passed = (Step("a", "A", ("x",)), "passed", 1.0)
    skipped = (Step("codeql", "CodeQL", ("x",)), "skipped", 0.0)

    assert "READY TO PUSH" in format_summary([passed])
    summary = format_summary([passed, skipped])
    assert "READY TO PUSH" not in summary
    assert "skipped and did not run: codeql" in summary
