"""Run each pytest file separately and print a per-file health report.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py tests/test_build_info.py tests/test_proof_routes.py
    python tests/run_all_tests.py --stop-on-fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TestFileResult:
    path: Path
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def discover_test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def resolve_requested_files(requested_files: list[str]) -> list[Path]:
    if not requested_files:
        return discover_test_files()

    resolved: list[Path] = []
    for raw_path in requested_files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path = path.resolve()

        try:
            path.relative_to(ROOT_DIR)
        except ValueError as exc:
            raise SystemExit(f"Refusing to run path outside repository: {raw_path}") from exc

        if not path.exists():
            raise SystemExit(f"Test file does not exist: {raw_path}")
        if not path.is_file() or path.name == Path(__file__).name:
            raise SystemExit(f"Not a runnable pytest file: {raw_path}")

        resolved.append(path)

    return resolved


def run_test_file(test_file: Path, extra_pytest_args: list[str]) -> TestFileResult:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file.relative_to(ROOT_DIR)),
        *extra_pytest_args,
    ]

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        errors="replace",
        check=False,
    )
    elapsed_seconds = time.perf_counter() - start

    return TestFileResult(
        path=test_file,
        returncode=completed.returncode,
        elapsed_seconds=elapsed_seconds,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def print_result(result: TestFileResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    relative_path = result.path.relative_to(ROOT_DIR)
    print(f"[{status}] {relative_path} ({result.elapsed_seconds:.1f}s)")

    if result.passed:
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
    if output.strip():
        print("-" * 80)
        print(output.rstrip())
        print("-" * 80)
    else:
        print(f"pytest exited with code {result.returncode} and no captured output.")


def print_summary(results: list[TestFileResult]) -> None:
    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]

    print("\n" + "=" * 80)
    print("TEST FILE SUMMARY")
    print("=" * 80)
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Total:  {len(results)}")

    if passed:
        print("\nWorking:")
        for result in passed:
            print(f"  - {result.path.relative_to(ROOT_DIR)}")

    if failed:
        print("\nIssues:")
        for result in failed:
            print(
                f"  - {result.path.relative_to(ROOT_DIR)} "
                f"(exit {result.returncode}, {result.elapsed_seconds:.1f}s)"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest files one by one and report which files pass or fail."
    )
    parser.add_argument(
        "test_files",
        nargs="*",
        help="Optional test files to run. Defaults to every tests/test_*.py file.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing test file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered test files without running them.",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Additional arguments passed to pytest, split on spaces.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_files = resolve_requested_files(args.test_files)

    if args.list:
        for test_file in test_files:
            print(test_file.relative_to(ROOT_DIR))
        return 0

    if not test_files:
        print("No tests/test_*.py files found.")
        return 1

    extra_pytest_args = args.pytest_args.split() if args.pytest_args else []
    results: list[TestFileResult] = []

    print(f"Running {len(test_files)} pytest file(s) separately...\n")
    for test_file in test_files:
        result = run_test_file(test_file, extra_pytest_args)
        results.append(result)
        print_result(result)

        if args.stop_on_fail and not result.passed:
            break

    print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
