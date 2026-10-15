"""Audit production frontend dependencies, retrying only failed registry requests.

`npm audit` exits 1 both when it finds vulnerabilities at the audit level and
when the registry's advisory endpoint fails ("audit endpoint returned an
error"). With ``--json`` the three outcomes differ: a finished audit reports
``metadata.vulnerabilities``; a failed advisory request reports a top-level
``message`` (npm/lib/utils/audit-error.js); any other npm error (for example
ENOLOCK) reports only ``error.code``. Vulnerabilities and other npm errors fail
at once; a failed advisory request is retried with backoff and fails only when
every attempt fails, so a registry outage is never reported as a clean audit.
``--offline=false`` overrides an offline npm config, which would otherwise skip
the request and report zero vulnerabilities.

Used by CI's frontend job and by ``npm run test:github``. Tool-only: nothing in
the MOTO runtime imports this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ARGS = ("audit", "--omit=dev", "--audit-level=high", "--offline=false")
ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (15, 45)
_FAILING_SEVERITIES = ("high", "critical")


def _json_payload(stdout: str) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(stdout or "")
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _finished_report(payload: Optional[dict[str, Any]]) -> bool:
    metadata = (payload or {}).get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("vulnerabilities"), dict)


def _request_failure(payload: Optional[dict[str, Any]]) -> Optional[str]:
    """Describe a failed advisory request, or None when npm failed for another reason."""
    message = (payload or {}).get("message")
    if not message:
        return None
    status = payload.get("statusCode")
    return f"{message} (HTTP {status})" if status else str(message)


def _other_npm_error(payload: Optional[dict[str, Any]], stderr: str) -> str:
    error = (payload or {}).get("error")
    if isinstance(error, dict) and error.get("code"):
        parts = [str(error.get(key) or "").strip() for key in ("code", "summary", "detail")]
        return " - ".join(part for part in parts if part)
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return "\n".join(lines[-5:]) or "npm printed no audit report"


def _describe_failing(report: dict[str, Any]) -> list[str]:
    lines = []
    for name, entry in sorted((report.get("vulnerabilities") or {}).items()):
        if not isinstance(entry, dict) or entry.get("severity") not in _FAILING_SEVERITIES:
            continue
        titles = sorted(
            {via.get("title", "") for via in entry.get("via") or [] if isinstance(via, dict)} - {""}
        )
        detail = "; ".join(titles) or "via " + ", ".join(
            via for via in entry.get("via") or [] if isinstance(via, str)
        )
        lines.append(f"  {entry['severity']:<8} {name} {entry.get('range', '')}: {detail}")
    return lines


def run_audit(
    cwd: Path,
    *,
    npm: Optional[str] = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    npm = npm or shutil.which("npm") or "npm"
    for attempt in range(1, ATTEMPTS + 1):
        result = run(
            [npm, *AUDIT_ARGS, "--json"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        report = _json_payload(result.stdout)
        if _finished_report(report):
            counts = report["metadata"]["vulnerabilities"]
            summary = ", ".join(f"{counts.get(level, 0)} {level}" for level in ("critical", "high", "moderate", "low"))
            if result.returncode == 0:
                print(f"npm audit passed ({summary}).", flush=True)
                return 0
            print(f"npm audit found vulnerabilities at or above high ({summary}):", flush=True)
            for line in _describe_failing(report):
                print(line, flush=True)
            return 1
        failure = _request_failure(report)
        if failure is None:
            print(f"npm audit failed: {_other_npm_error(report, result.stderr)}", flush=True)
            return 1
        print(
            f"npm audit could not reach the advisory service (attempt {attempt}/{ATTEMPTS}): {failure}",
            flush=True,
        )
        if attempt < ATTEMPTS:
            sleep(RETRY_DELAYS_SECONDS[attempt - 1])
    print(
        f"npm audit failed: the advisory service did not answer in {ATTEMPTS} attempts, "
        "so the dependencies were not audited.",
        flush=True,
    )
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cwd",
        type=Path,
        default=REPO_ROOT / "frontend",
        help="Directory holding package-lock.json (default: frontend).",
    )
    args = parser.parse_args(argv)
    return run_audit(args.cwd.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
