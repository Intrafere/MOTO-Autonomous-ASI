import json
from pathlib import Path
import subprocess
import sys

from scripts.run_npm_audit import ATTEMPTS, AUDIT_ARGS, RETRY_DELAYS_SECONDS, run_audit


def _report(**counts: int) -> str:
    levels = {"info": 0, "low": 0, "moderate": 0, "high": 0, "critical": 0}
    levels.update(counts)
    vulnerabilities = {}
    if counts.get("high"):
        vulnerabilities["left-pad"] = {
            "severity": "high",
            "range": "<1.3.0",
            "via": [{"title": "Prototype pollution in left-pad"}],
        }
    return json.dumps(
        {
            "auditReportVersion": 2,
            "vulnerabilities": vulnerabilities,
            "metadata": {"vulnerabilities": {**levels, "total": sum(levels.values())}},
        }
    )


# What npm prints with --json when the advisory endpoint request fails
# (npm/lib/utils/audit-error.js); the process exits 1 like a failing audit.
_ENDPOINT_FAILURE = json.dumps(
    {
        "message": "503 Service Unavailable - POST https://registry.npmjs.org/-/npm/v1/security/audits/quick",
        "method": "POST",
        "uri": "https://registry.npmjs.org/-/npm/v1/security/audits/quick",
        "statusCode": 503,
        "body": "",
    }
)


class _FakeNpm:
    def __init__(self, *outcomes: tuple[int, str]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        returncode, stdout = self.outcomes.pop(0)
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")


def _audit(fake: _FakeNpm, sleeps: list[float]) -> int:
    return run_audit(Path("frontend"), npm="npm", run=fake, sleep=sleeps.append)


def test_clean_audit_passes_in_one_call(capsys) -> None:
    fake, sleeps = _FakeNpm((0, _report(moderate=1))), []
    assert _audit(fake, sleeps) == 0
    # An offline npm config would skip the request and report zero findings.
    assert "--offline=false" in AUDIT_ARGS
    assert fake.calls == [["npm", *AUDIT_ARGS, "--json"]]
    assert sleeps == []
    assert "npm audit passed (0 critical, 0 high, 1 moderate, 0 low)" in capsys.readouterr().out


def test_vulnerabilities_fail_at_once_without_retry(capsys) -> None:
    fake, sleeps = _FakeNpm((1, _report(high=1))), []
    assert _audit(fake, sleeps) == 1
    assert len(fake.calls) == 1 and sleeps == []
    out = capsys.readouterr().out
    assert "found vulnerabilities at or above high" in out
    assert "left-pad <1.3.0: Prototype pollution in left-pad" in out


def test_failed_registry_request_is_retried_until_the_audit_finishes(capsys) -> None:
    fake, sleeps = _FakeNpm((1, _ENDPOINT_FAILURE), (0, _report())), []
    assert _audit(fake, sleeps) == 0
    assert len(fake.calls) == 2
    assert sleeps == [RETRY_DELAYS_SECONDS[0]]
    assert "could not reach the advisory service (attempt 1/3): 503 Service Unavailable" in capsys.readouterr().out


def test_registry_that_never_answers_fails_instead_of_passing(capsys) -> None:
    fake = _FakeNpm(*[(1, _ENDPOINT_FAILURE)] * ATTEMPTS)
    sleeps: list[float] = []
    assert _audit(fake, sleeps) == 1
    assert len(fake.calls) == ATTEMPTS
    assert sleeps == list(RETRY_DELAYS_SECONDS[: ATTEMPTS - 1])
    assert "were not audited" in capsys.readouterr().out


def test_other_npm_errors_fail_at_once_with_npm_reason(capsys) -> None:
    # npm's --json output without a package-lock.json; retrying cannot fix it.
    no_lockfile = json.dumps(
        {"error": {"code": "ENOLOCK", "summary": "This command requires an existing lockfile.", "detail": ""}}
    )
    fake, sleeps = _FakeNpm((1, no_lockfile)), []
    assert _audit(fake, sleeps) == 1
    assert len(fake.calls) == 1 and sleeps == []
    out = capsys.readouterr().out
    assert "npm audit failed: ENOLOCK - This command requires an existing lockfile." in out
    assert "advisory service" not in out


def test_utf8_report_is_decoded_as_utf8_on_every_platform(capsys) -> None:
    # Advisory text such as U+201D is not decodable as Windows cp1252.
    report = json.loads(_report(moderate=1))
    report["vulnerabilities"]["quote-lib"] = {
        "severity": "moderate",
        "range": "<2.0.0",
        "via": [{"title": "Unescaped “quotes” in quote-lib"}],
    }
    payload = json.dumps(report, ensure_ascii=False).encode("utf-8")
    assert b"\xe2\x80\x9d" in payload
    child = f"import sys; sys.stdout.buffer.write({payload!r})"

    def run_real_child(args, **kwargs):
        return subprocess.run([sys.executable, "-c", child], **kwargs)

    assert run_audit(Path("."), npm="npm", run=run_real_child, sleep=lambda _: None) == 0
    assert "npm audit passed" in capsys.readouterr().out
