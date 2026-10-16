"""Run GitHub's CodeQL code scanning locally and fail on alerts main does not have.

GitHub's CodeQL default setup only analyzes pull requests into main, so a dev
branch can pass every test and still fail code scanning at pull-request time.
This runs the same analyses (actions, python, javascript-typescript; build mode
none) with the security-extended suite, a superset of default setup's queries,
on a clean export of the tree about to be pushed. Every alert is compared with
the same scan of origin/main the way a pull-request check does, so only alerts
the branch introduces fail the run.

Tool-only: nothing in the MOTO runtime imports this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
# Bundle installed by --install: the github/codeql-action release default setup
# used when this was last updated. Bump it when GitHub moves to a newer bundle.
CODEQL_BUNDLE_VERSION = "2.27.2"
LANGUAGES = ("actions", "python", "javascript")
DEFAULT_BASELINE = "origin/main"
# Bump when the stored result shape or the alert matching changes.
RESULTS_FORMAT = 1
# Index filters from CodeQL 2.27.2 actions/tools/autobuild.sh; see run_actions_autobuild.
ACTIONS_PATH_FILTERS = (
    "exclude:**/*",
    "include:.github/workflows/*.yml",
    "include:.github/workflows/*.yaml",
    "include:.github/reusable_workflows/**/*.yml",
    "include:.github/reusable_workflows/**/*.yaml",
    "include:**/action.yml",
    "include:**/action.yaml",
    "include:**/actions.lock",
)
_ACTIONS_EXTRACTOR_DIRS = (
    "DIAGNOSTIC_DIR",
    "LOG_DIR",
    "SCRATCH_DIR",
    "SOURCE_ARCHIVE_DIR",
    "TRAP_DIR",
    "WIP_DATABASE",
)


class CodeQLRunError(RuntimeError):
    """Raised when CodeQL or Git cannot complete the scan."""


@dataclass(frozen=True)
class Alert:
    language: str
    rule: str
    severity: str
    path: str
    line: int
    message: str
    source_line: str
    default_suite: bool

    def match_key(self) -> tuple[str, str, str, str]:
        """Identity that survives unrelated edits, like GitHub's line-hash fingerprint."""
        return (self.rule, self.path, self.message, self.source_line)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _codeql_executable_name() -> str:
    return "codeql.exe" if _is_windows() else "codeql"


def tools_root() -> Path:
    """Folder holding installed CodeQL bundles, one subfolder per version."""
    if _is_windows():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "codeql"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "codeql"


def default_work_dir() -> Path:
    """Scratch space for databases plus the per-tree results cache."""
    if os.environ.get("RUNNER_TEMP"):
        return Path(os.environ["RUNNER_TEMP"]) / "moto-codeql"
    if _is_windows():
        # The Python extractor runs through `py -3`, which may be a Microsoft
        # Store Python that only sees a private copy of anything it writes under
        # %LOCALAPPDATA%; the temp folder is not virtualized.
        return Path(tempfile.gettempdir()) / "moto-codeql-scan"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "moto" / "codeql-scan"


def _version_key(path: Path) -> tuple[int, ...]:
    for part in reversed(path.parts):
        match = re.search(r"\d+(?:\.\d+)+", part)
        if match:
            return tuple(int(piece) for piece in match.group(0).split("."))
    return ()


def find_codeql(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate a CodeQL CLI: explicit path, MOTO_CODEQL, installed bundles, runner cache, PATH."""
    exe = _codeql_executable_name()
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates: list[Path] = []
    if os.environ.get("MOTO_CODEQL"):
        candidates.append(Path(os.environ["MOTO_CODEQL"]))
    root = tools_root()
    candidates.append(root / f"v{CODEQL_BUNDLE_VERSION}" / "codeql" / exe)
    candidates.extend(sorted(root.glob(f"v*/codeql/{exe}"), key=_version_key, reverse=True))
    if os.environ.get("RUNNER_TOOL_CACHE"):
        toolcache = Path(os.environ["RUNNER_TOOL_CACHE"]) / "CodeQL"
        candidates.extend(sorted(toolcache.glob(f"*/x64/codeql/{exe}"), key=_version_key, reverse=True))
    on_path = shutil.which("codeql")
    if on_path:
        candidates.append(Path(on_path))
    return next((path for path in candidates if path.is_file()), None)


def _bundle_asset() -> str:
    if _is_windows():
        return "codeql-bundle-win64.tar.gz"
    if sys.platform == "darwin":
        return "codeql-bundle-osx64.tar.gz"
    return "codeql-bundle-linux64.tar.gz"


def _extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    if not hasattr(tarfile, "data_filter"):
        raise CodeQLRunError("Safe tar extraction needs Python 3.11.4 or newer.")
    archive.extractall(destination, filter="data")


def install_codeql() -> Path:
    """Download the pinned CodeQL bundle from github/codeql-action and verify its checksum."""
    sys.path.insert(0, str(REPO_ROOT))
    from launcher_https import verified_urlopen

    asset = _bundle_asset()
    base_url = (
        "https://github.com/github/codeql-action/releases/download/"
        f"codeql-bundle-v{CODEQL_BUNDLE_VERSION}/{asset}"
    )
    target = tools_root() / f"v{CODEQL_BUNDLE_VERSION}"
    executable = target / "codeql" / _codeql_executable_name()
    if executable.is_file():
        return executable
    target.mkdir(parents=True, exist_ok=True)
    with verified_urlopen(f"{base_url}.checksum.txt", timeout=60) as response:
        expected = response.read().decode("utf-8").split()[0].lower()
    partial = target / f"{asset}.partial"
    digest = hashlib.sha256()
    print(f"Downloading CodeQL {CODEQL_BUNDLE_VERSION} ({asset}) ...", flush=True)
    with verified_urlopen(base_url, timeout=600) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise CodeQLRunError(f"Checksum mismatch for {asset}; the download was discarded.")
    print("Extracting CodeQL bundle ...", flush=True)
    with tarfile.open(partial, "r:gz") as archive:
        _extract_tar(archive, target)
    partial.unlink()
    if not executable.is_file():
        raise CodeQLRunError(f"The CodeQL bundle did not contain {executable}.")
    return executable


def _git(*args: str, repo: Path, env: Optional[dict[str, str]] = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CodeQLRunError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def snapshot_tree(rev: Optional[str], repo: Path = REPO_ROOT) -> tuple[str, str]:
    """Return (tree id, description) for a revision or for the working tree.

    The working tree snapshot is exactly what ``git add -A`` would commit,
    including new untracked files, built in a throwaway index so the real
    index is never touched.
    """
    if rev:
        return _git("rev-parse", f"{rev}^{{tree}}", repo=repo), f"{rev} ({_git('rev-parse', '--short', rev, repo=repo)})"
    index_path = Path(_git("rev-parse", "--git-path", "index", repo=repo))
    if not index_path.is_absolute():
        index_path = repo / index_path
    with tempfile.TemporaryDirectory() as scratch:
        temp_index = Path(scratch) / "index"
        if index_path.is_file():
            shutil.copyfile(index_path, temp_index)
        env = {**os.environ, "GIT_INDEX_FILE": str(temp_index)}
        _git("add", "-A", repo=repo, env=env)
        tree = _git("write-tree", repo=repo, env=env)
    head_tree = _git("rev-parse", "HEAD^{tree}", repo=repo)
    if tree == head_tree:
        return tree, f"HEAD ({_git('rev-parse', '--short', 'HEAD', repo=repo)})"
    return tree, "working tree (uncommitted and untracked changes included)"


def export_tree(tree: str, destination: Path, repo: Path = REPO_ROOT) -> None:
    """Write the files of ``tree`` into ``destination``, as a fresh checkout would."""
    destination.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", tree],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        _extract_tar(archive, destination)
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.wait() != 0:
        raise CodeQLRunError(f"git archive {tree} failed: {stderr.strip()}")


def _run_codeql(codeql: Path, args: Sequence[str], *, log: Path, cwd: Optional[Path] = None) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ codeql {' '.join(args)}\n")
        handle.flush()
        result = subprocess.run(
            [str(codeql), *args],
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise CodeQLRunError(f"codeql {args[0]} {args[1]} failed (see {log}):\n" + "\n".join(tail))


def codeql_version(codeql: Path) -> str:
    result = subprocess.run(
        [str(codeql), "version", "--format=json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CodeQLRunError(f"Could not run {codeql}: {result.stderr.strip()}")
    return str(json.loads(result.stdout)["version"])


def build_database(codeql: Path, language: str, source: Path, database: Path, log: Path) -> None:
    if language == "actions" and _is_windows():
        # The bundled Windows actions autobuild is a PowerShell script that
        # execution policies commonly block; trace the same work through
        # run_actions_autobuild instead of changing the policy.
        _run_codeql(
            codeql,
            ["database", "init", str(database), "--language=actions", f"--source-root={source}", "--overwrite"],
            log=log,
        )
        _run_codeql(
            codeql,
            [
                "database",
                "trace-command",
                str(database),
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "--actions-autobuild",
            ],
            log=log,
            cwd=source,
        )
        _run_codeql(codeql, ["database", "finalize", str(database), "--threads=0"], log=log)
        return
    _run_codeql(
        codeql,
        [
            "database",
            "create",
            str(database),
            f"--language={language}",
            "--build-mode=none",
            f"--source-root={source}",
            "--threads=0",
            "--overwrite",
        ],
        log=log,
    )


def run_actions_autobuild() -> int:
    """Port of CodeQL 2.27.2 actions/tools/autobuild.sh, run under ``trace-command``.

    Points the JavaScript extractor at workflow and action files only.
    """
    user_filters = os.environ.get("LGTM_INDEX_FILTERS")
    filters = "\n".join(ACTIONS_PATH_FILTERS)
    os.environ["LGTM_INDEX_FILTERS"] = f"{filters}\n{user_filters}" if user_filters else filters
    codeql = Path(os.environ["CODEQL_DIST"]) / _codeql_executable_name()
    resolved = subprocess.run(
        [str(codeql), "resolve", "extractor", "--language", "javascript"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    javascript_root = resolved[-1].strip()
    os.environ["CODEQL_EXTRACTOR_JAVASCRIPT_ROOT"] = javascript_root
    for suffix in _ACTIONS_EXTRACTOR_DIRS:
        os.environ[f"CODEQL_EXTRACTOR_JAVASCRIPT_{suffix}"] = os.environ[f"CODEQL_EXTRACTOR_ACTIONS_{suffix}"]
    tools = Path(javascript_root) / "tools"
    if _is_windows():
        return subprocess.call(["cmd.exe", "/c", str(tools / "autobuild.cmd")])
    return subprocess.call([str(tools / "autobuild.sh")])


def default_suite_rule_ids(codeql: Path, language: str) -> set[str]:
    """Rule IDs in the suite default setup runs, so reports can say what GitHub flags."""
    result = subprocess.run(
        [str(codeql), "resolve", "queries", f"codeql/{language}-queries:codeql-suites/{language}-code-scanning.qls"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CodeQLRunError(f"Could not resolve the {language} default suite: {result.stderr.strip()}")
    rule_ids = set()
    for line in result.stdout.splitlines():
        query = Path(line.strip())
        if query.suffix != ".ql" or not query.is_file():
            continue
        match = re.search(r"@id\s+(\S+)", query.read_text(encoding="utf-8", errors="replace"))
        if match:
            rule_ids.add(match.group(1))
    return rule_ids


def _severity_label(rule: dict, result: dict) -> str:
    properties = rule.get("properties", {})
    score = properties.get("security-severity")
    if score is not None:
        value = float(score)
        tier = "critical" if value >= 9 else "high" if value >= 7 else "medium" if value >= 4 else "low"
        return f"{tier} ({value:g})"
    return str(result.get("level") or rule.get("defaultConfiguration", {}).get("level") or "warning")


def load_alerts(sarif_path: Path, language: str, source_root: Path, default_rules: Iterable[str]) -> list[Alert]:
    default_rules = set(default_rules)
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    alerts = []
    for run in sarif.get("runs", []):
        rules = {}
        for component in [run["tool"]["driver"], *run["tool"].get("extensions", [])]:
            for rule in component.get("rules", []):
                rules[rule["id"]] = rule
        for result in run.get("results", []):
            rule_id = result["ruleId"]
            location = result["locations"][0]["physicalLocation"]
            path = location["artifactLocation"]["uri"]
            line = int(location.get("region", {}).get("startLine", 0))
            try:
                source_line = (
                    (source_root / path).read_text(encoding="utf-8", errors="replace").splitlines()[line - 1].strip()
                )
            except (OSError, IndexError):
                source_line = ""
            # The first line names the problem; later lines list data-flow
            # sources, whose number shifts with unrelated edits.
            message = re.sub(r"\]\(\d+\)", "]", result["message"]["text"]).splitlines()[0]
            alerts.append(
                Alert(
                    language=language,
                    rule=rule_id,
                    severity=_severity_label(rules.get(rule_id, {}), result),
                    path=path,
                    line=line,
                    message=message,
                    source_line=source_line,
                    default_suite=rule_id in default_rules,
                )
            )
    return alerts


def new_alerts(candidate: Sequence[Alert], baseline: Sequence[Alert]) -> list[Alert]:
    """Candidate alerts with no matching baseline alert, counting duplicates."""
    remaining = Counter(alert.match_key() for alert in baseline)
    introduced = []
    for alert in sorted(candidate, key=lambda item: (item.path, item.line, item.rule)):
        key = alert.match_key()
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            introduced.append(alert)
    return introduced


def scan_tree(
    tree: str,
    *,
    codeql: Path,
    version: str,
    work_dir: Path,
    label: str,
    keep: bool,
    repo: Path = REPO_ROOT,
) -> tuple[list[Alert], bool]:
    """Return (alerts, from_cache) for ``tree``, reusing results for an identical tree."""
    cache = work_dir / "results" / f"{tree}-codeql-{version}-v{RESULTS_FORMAT}.json"
    if cache.is_file():
        return [Alert(**item) for item in json.loads(cache.read_text(encoding="utf-8"))], True
    scan_dir = work_dir / "scans" / tree[:16]
    shutil.rmtree(scan_dir, ignore_errors=True)
    source = scan_dir / "src"
    export_tree(tree, source, repo=repo)
    alerts: list[Alert] = []
    for language in LANGUAGES:
        started = time.monotonic()
        print(f"[{label}] {language}: building database and analyzing ...", flush=True)
        database = scan_dir / f"db-{language}"
        sarif = scan_dir / f"{language}.sarif"
        log = scan_dir / f"{language}.log"
        build_database(codeql, language, source, database, log)
        _run_codeql(
            codeql,
            [
                "database",
                "analyze",
                str(database),
                f"codeql/{language}-queries:codeql-suites/{language}-security-extended.qls",
                "--format=sarif-latest",
                f"--sarif-category=/language:{language}",
                f"--output={sarif}",
                "--threads=0",
            ],
            log=log,
        )
        found = load_alerts(sarif, language, source, default_suite_rule_ids(codeql, language))
        alerts.extend(found)
        print(f"[{label}] {language}: {len(found)} alert(s) in {time.monotonic() - started:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(alert) for alert in alerts], indent=1), encoding="utf-8")
    if not keep:
        shutil.rmtree(scan_dir, ignore_errors=True)
    return alerts, False


def _fetch_baseline(baseline: str, repo: Path) -> None:
    remote, _, branch = baseline.partition("/")
    if not branch:
        return
    result = subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", "--no-tags", remote, branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"WARNING: could not fetch {baseline}; comparing with the local copy. "
            f"{(result.stderr or result.stdout).strip()}",
            flush=True,
        )


def format_alert(alert: Alert) -> str:
    who = "GitHub default setup reports this" if alert.default_suite else "security-extended suite only"
    return (
        f"  NEW {alert.severity:<14} {alert.rule}  {alert.path}:{alert.line}  [{who}]\n"
        f"      {alert.message}\n"
        f"      > {alert.source_line[:200]}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run GitHub's CodeQL analyses (actions, python, javascript) locally and fail "
            "on any alert the baseline branch does not already have."
        )
    )
    parser.add_argument("--rev", help="Scan this commit instead of the working tree.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Branch to compare with (default: origin/main).")
    parser.add_argument("--no-fetch", action="store_true", help="Do not fetch the baseline branch first.")
    parser.add_argument("--codeql", help="Path to the codeql executable.")
    parser.add_argument("--install", action="store_true", help=f"Install CodeQL {CODEQL_BUNDLE_VERSION} and exit.")
    parser.add_argument("--install-if-missing", action="store_true", help="Install CodeQL when none is found.")
    parser.add_argument("--work-dir", type=Path, help="Scratch and results-cache folder.")
    parser.add_argument("--keep", action="store_true", help="Keep exported sources and databases.")
    parser.add_argument("--actions-autobuild", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.actions_autobuild:
        return run_actions_autobuild()
    try:
        if args.install:
            print(f"CodeQL ready: {install_codeql()}")
            return 0
        codeql = find_codeql(args.codeql)
        if codeql is None and args.install_if_missing and not args.codeql:
            codeql = install_codeql()
        if codeql is None:
            print(
                "CodeQL CLI not found. Run `python -m scripts.run_codeql --install` "
                f"(downloads the {CODEQL_BUNDLE_VERSION} bundle from github/codeql-action) "
                "or pass --codeql / set MOTO_CODEQL.",
                file=sys.stderr,
            )
            return 2
        version = codeql_version(codeql)
        work_dir = (args.work_dir or default_work_dir()).resolve()
        print(f"CodeQL {version}: {codeql}", flush=True)
        if not args.no_fetch:
            _fetch_baseline(args.baseline, REPO_ROOT)
        candidate_tree, candidate_name = snapshot_tree(args.rev)
        baseline_tree, baseline_name = snapshot_tree(args.baseline)
        common = {"codeql": codeql, "version": version, "work_dir": work_dir, "keep": args.keep}
        candidate, candidate_cached = scan_tree(candidate_tree, label="candidate", **common)
        baseline, baseline_cached = scan_tree(baseline_tree, label="baseline", **common)
    except CodeQLRunError as exc:
        print(f"CodeQL scan failed: {exc}", file=sys.stderr)
        return 2

    introduced = new_alerts(candidate, baseline)
    print()
    for title, name, tree, alerts, cached in (
        ("Candidate", candidate_name, candidate_tree, candidate, candidate_cached),
        ("Baseline", baseline_name, baseline_tree, baseline, baseline_cached),
    ):
        reuse = " (cached result)" if cached else ""
        print(f"{title + ':':<10} {name}, tree {tree[:12]}: {len(alerts)} alert(s){reuse}")
    if not introduced:
        print("PASS: no CodeQL alerts beyond those already on the baseline.")
        return 0
    print(f"FAIL: {len(introduced)} new CodeQL alert(s) the baseline does not have:")
    for alert in introduced:
        print(format_alert(alert))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
