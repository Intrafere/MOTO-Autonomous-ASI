"""Run the deterministic deep workflow suite on Windows and Unix-like systems."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    env = os.environ.copy()
    env["MOTO_WORKFLOW_DEEP_TESTS"] = "1"
    model_result = subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/workflow_scenarios",
            "tests/workflow_recombinatory",
            "tests/workflow_cross_field",
        ],
        cwd=ROOT_DIR,
        env=env,
    )
    if model_result:
        return model_result

    env["MOTO_REAL_ADAPTER_DEEP_TESTS"] = "1"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/workflow_real_adapters/test_build_f_deep_matrix.py",
        ],
        cwd=ROOT_DIR,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
