"""Run every check GitHub runs on this repository, locally, before pushing.

Mirrors .github/workflows/ci.yml (Python tests, Windows native lifecycle, and
frontend build jobs), the SyntheticLib release-evidence workflow (including its pytest-only
test environment), and CodeQL code scanning, which GitHub's default setup only
runs on pull requests into main. Every step runs even after a failure, and the
summary names each step that passed, failed, or was skipped.

Not mirrored: CI's clean ``npm ci`` install; the frontend steps use the existing
frontend/node_modules.

Tool-only: nothing in the MOTO runtime imports this module.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETICLIB_EVIDENCE = (
    "syntheticlib-trade/compatibility/compatibility-matrix.json",
    "syntheticlib-trade/compatibility/moto-contract-test-result.json",
    "syntheticlib-trade/compatibility/moto-compatibility-receipt.json",
    "syntheticlib-trade/compatibility/moto-compatibility-receipt.sha256",
    "syntheticlib-trade/compatibility/frozen-asset-manifest.json",
    "syntheticlib-trade/compatibility/compatibility-bundle-manifest.json",
    "syntheticlib-trade/compatibility/moto-preflight-attestation.json",
    "syntheticlib-trade/compatibility/MOTO_COMPATIBILITY_RECEIPT.md",
)
# The release-evidence workflow installs only pytest before running these.
RELEASE_FOCUSED_TESTS = (
    "tests/unit/test_syntheticlib_trade_compatibility.py",
    "tests/integration/test_syntheticlib_trade_contract_command.py",
    "tests/unit/test_local_syntheticlib_compatibility_ci.py",
)
# Runs pytest as if only pytest were installed: every other installed package
# (and every pytest plugin) is hidden, so a test-suite import of a runtime
# dependency fails here the way it fails on the release runner.
PYTEST_ONLY_BOOTSTRAP = """
import importlib.abc
from importlib import metadata
import os
import site
import sys
import sysconfig

from packaging.requirements import Requirement


def _normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def _pytest_install_set():
    pending, installed = ["pytest"], set()
    while pending:
        name = pending.pop()
        if _normalize(name) in installed:
            continue
        installed.add(_normalize(name))
        for raw in metadata.requires(name) or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return installed


_INSTALLED = _pytest_install_set()
_ALLOWED = {
    top
    for top, distributions in metadata.packages_distributions().items()
    if any(_normalize(dist) in _INSTALLED for dist in distributions)
}
_SITE_DIRS = tuple(
    os.path.normcase(os.path.abspath(path)) + os.sep
    for path in {
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
        site.getusersitepackages(),
    }
)


class _OnlyPytestInstalled(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in _ALLOWED:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is None:
                continue
            locations = [spec.origin or "", *(spec.submodule_search_locations or [])]
            if any(
                os.path.normcase(os.path.abspath(location)).startswith(_SITE_DIRS)
                for location in locations
                if location and location not in ("built-in", "frozen")
            ):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None
        return None


sys.meta_path.insert(0, _OnlyPytestInstalled())
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
"""


@dataclass(frozen=True)
class Step:
    slug: str
    title: str
    command: tuple[str, ...]
    cwd: str = "."


def build_steps(python: str = sys.executable, npm: Optional[str] = None) -> tuple[Step, ...]:
    npm = npm or shutil.which("npm") or "npm"
    return (
        Step("pip-check", "Python dependency consistency", (python, "-m", "pip", "check")),
        Step(
            "artifacts",
            "Workflow artifact projection",
            (python, "-m", "tests.workflow_cross_field.artifacts", "--check"),
        ),
        Step(
            "syntheticlib-tracked",
            "SyntheticLib evidence files are tracked",
            ("git", "ls-files", "--error-unmatch", *SYNTHETICLIB_EVIDENCE),
        ),
        Step(
            "syntheticlib-contract",
            "SyntheticLib compatibility contract",
            (python, "-m", "scripts.check_syntheticlib_trade_contract", "--check"),
        ),
        Step("pytest", "Python tests", (python, "-m", "pytest")),
        Step(
            "npm-audit",
            "Frontend production dependency audit",
            (npm, "audit", "--omit=dev", "--audit-level=high"),
            cwd="frontend",
        ),
        Step("vitest", "Frontend tests", (npm, "run", "test"), cwd="frontend"),
        Step("build", "Frontend build", (npm, "run", "build"), cwd="frontend"),
        Step("browser", "Browser smoke tests", (npm, "run", "test:browser"), cwd="frontend"),
        Step(
            "release-tests",
            "SyntheticLib release tests with only pytest installed",
            (python, "-c", PYTEST_ONLY_BOOTSTRAP, *RELEASE_FOCUSED_TESTS),
        ),
        Step(
            "release-evidence",
            "SyntheticLib release evidence",
            (python, "-m", "scripts.run_local_syntheticlib_compatibility_ci"),
        ),
        Step(
            "codeql",
            "CodeQL: no new alerts vs origin/main",
            (python, "-m", "scripts.run_codeql"),
        ),
    )


def run_steps(
    steps: Sequence[Step],
    *,
    skip: Sequence[str] = (),
    repo: Path = REPO_ROOT,
) -> list[tuple[Step, str, float]]:
    """Run each step with live output; return (step, status, seconds) in order."""
    outcomes = []
    for step in steps:
        if step.slug in skip:
            outcomes.append((step, "skipped", 0.0))
            continue
        print(f"\n=== [{step.slug}] {step.title}", flush=True)
        started = time.monotonic()
        try:
            returncode = subprocess.run(step.command, cwd=repo / step.cwd, check=False).returncode
        except OSError as exc:
            print(f"Could not start {step.command[0]}: {exc}", flush=True)
            returncode = 1
        status = "passed" if returncode == 0 else f"FAILED (exit {returncode})"
        outcomes.append((step, status, time.monotonic() - started))
    return outcomes


def format_summary(outcomes: Sequence[tuple[Step, str, float]]) -> str:
    lines = ["", "GitHub check parity summary:"]
    for step, status, seconds in outcomes:
        duration = f"{seconds:6.0f}s" if status != "skipped" else "      -"
        lines.append(f"  {status:<18} {duration}  {step.title} [{step.slug}]")
    failed = [step.slug for step, status, _ in outcomes if status.startswith("FAILED")]
    skipped = [step.slug for step, status, _ in outcomes if status == "skipped"]
    if failed:
        lines.append(f"NOT READY TO PUSH: {len(failed)} check(s) failed: {', '.join(failed)}")
    elif skipped:
        lines.append(f"Passed, but these checks were skipped and did not run: {', '.join(skipped)}")
    else:
        lines.append("READY TO PUSH: every check GitHub runs passed locally.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    steps = build_steps()
    slugs = [step.slug for step in steps]
    parser = argparse.ArgumentParser(
        description="Run every check GitHub runs (ci.yml, release evidence, CodeQL) before pushing."
    )
    parser.add_argument("--skip", action="append", default=[], choices=slugs, help="Skip a check (repeatable).")
    parser.add_argument("--only", action="append", default=[], choices=slugs, help="Run only these checks.")
    parser.add_argument("--list", action="store_true", help="List the checks and exit.")
    args = parser.parse_args(argv)
    if args.list:
        for step in steps:
            print(f"{step.slug:<22} {step.title}")
        return 0
    skip = set(args.skip)
    if args.only:
        skip |= set(slugs) - set(args.only)
    outcomes = run_steps(steps, skip=sorted(skip))
    print(format_summary(outcomes))
    return 1 if any(status.startswith("FAILED") for _, status, _ in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
