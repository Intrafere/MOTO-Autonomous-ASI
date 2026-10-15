from pathlib import Path
import re
import sys

from scripts.run_github_checks import (
    REPO_ROOT,
    SYNTHETICLIB_EVIDENCE,
    Step,
    build_steps,
    format_summary,
    run_steps,
)


CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


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
        "npm audit --omit=dev --audit-level=high",
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
        "npm audit --omit=dev --audit-level=high",
        "npm run test",
        "npm run build",
        "npm run test:browser",
    } <= local


def test_syntheticlib_evidence_list_matches_ci() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    block = workflow.split("git ls-files --error-unmatch", 1)[1].split("python -m", 1)[0]
    assert tuple(re.findall(r"syntheticlib-trade/compatibility/\S+?(?=\s|\\|$)", block)) == SYNTHETICLIB_EVIDENCE


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
