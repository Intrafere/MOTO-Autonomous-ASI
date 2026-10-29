"""Refuse to publish private plans, runtime data, or credentials in a git push.

``.gitignore`` only keeps untracked files out of ``git add -A``. It cannot stop
a file that was already committed (forced with ``git add -f``, or re-included by
a ``!`` negation), and GitHub keeps every pushed commit downloadable by SHA even
after a force-push removes it from the branch. So this checks exactly what a
push would upload: every commit, tag, and blob reachable from the pushed refs
that the push URL (or a configured public reference remote) does not already
advertise on a branch or tag. Pull-request refs do not count, so purged content
cannot be re-published through them, and local remote-tracking refs never count,
since a stale one can still point at a commit the remote has dropped.

A push fails when it would upload:
- a path that is private (planning notes, proof-trade material outside the
  published compatibility evidence, MOTO runtime data from any data root,
  environment and key files, dev logs, local tool state) or ignored by the
  newest .gitignore, whether the file's content is new or not, or an embedded
  repository .gitmodules does not declare;
- a binary file other than an image, which cannot be read here;
- a line of a new or changed file, a commit message, or a tag message holding a
  credential-shaped value, a user home-directory path, or a personal email that
  is not already public;
- text or distinctive details (endpoint paths and hosts; regions and rates only
  alongside other private text) from the private planning files present in this
  clone; clones without those files skip this part;
- a commit or tag by a personal webmail address that is not already public.

Install once per clone with ``python -m scripts.check_publish_safety --install``
(``npm run test:github`` reinstalls it on every run). That copies this file and
a snapshot of .gitignore into the clone's shared hooks directory, so the check
runs from every branch and worktree, including old checkouts that predate it. An
existing pre-push hook (for example Git LFS) is kept and run first with the same
input. Pushes to remotes named in ``git config moto.publishSafety.privateRemotes``
are allowed and logged; ``moto.publishSafety.publicRemotes`` names remotes whose
branches and tags also count as public (for example ``upstream`` for a fork); a
remote's name alone never makes it public.
Self-contained (standard library only) because the installed copy runs outside
the checkout. Tool-only: nothing in the MOTO runtime imports this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Iterable, Optional, Sequence, TextIO


HOOK_MARKER = "moto-publish-safety"
INSTALLED_NAME = "moto_publish_safety.py"
GITIGNORE_SNAPSHOT_NAME = "moto_publish_safety.gitignore"
CHAINED_HOOK_NAME = "pre-push.moto-chained"
ZERO_SHA = "0" * 40
GIT_TIMEOUT_SECONDS = 120
LS_REMOTE_TIMEOUT_SECONDS = 30
MAX_PRIVATE_SOURCE_BYTES = 5 * 1024 * 1024
WINDOW_WORDS = 10
# Consecutive matching 10-word windows (about 17 shared words) before text counts
# as copied, so a one-word edit near a benign shared phrase does not trip it.
MIN_WINDOW_RUN = 8
# Read git objects as stored: replace refs and grafts change what git shows, not
# what git push uploads.
_GIT_ENV = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull}

PUBLISHABLE_TRADE_FILES = frozenset({
    "syntheticlib-trade/compatibility/README.md",
    "syntheticlib-trade/compatibility/compatibility-matrix.json",
    "syntheticlib-trade/compatibility/moto-contract-test-result.json",
    "syntheticlib-trade/compatibility/moto-compatibility-receipt.json",
    "syntheticlib-trade/compatibility/moto-compatibility-receipt.sha256",
    "syntheticlib-trade/compatibility/frozen-asset-manifest.json",
    "syntheticlib-trade/compatibility/compatibility-bundle-manifest.json",
    "syntheticlib-trade/compatibility/moto-preflight-attestation.json",
    "syntheticlib-trade/compatibility/MOTO_COMPATIBILITY_RECEIPT.md",
})
TRACKED_PLACEHOLDERS = frozenset({
    "backend/data/.gitkeep",
    "backend/data/auto_final_answer/.gitkeep",
    "backend/data/proofs/.gitkeep",
    "backend/data/user_uploads/.gitkeep",
    "backend/logs/.gitkeep",
})
_PRIVATE_FOLDERS = frozenset({"plans", "(old) syntheticlib upgrade plans", ".moto_instances"})
# Folders and files a MOTO data root holds, recognised under any data root.
_RUNTIME_FOLDERS = frozenset({
    "auto_sessions", "auto_brainstorms", "auto_papers", "auto_final_answer", "manual_proofs",
    "manual_proof_runs", "leanoj_sessions", "leanoj_artifacts", "leanoj_partial_proofs", "chroma_db",
    "solution_paths", "syntheticlib_trade", "lean4_workspace", "user_uploads",
})
_RUNTIME_FILES = frozenset({
    "runtime_settings.json", "provider_notifications.json", "rag_shared_training.txt",
    "aggregator_event_log.txt", "aggregator_stats.json", "auto_research_metadata.json",
    "auto_research_stats.json", "auto_workflow_state.json", "auto_research_topic_rejections.txt",
    "proofs_index.json", "manual_compiler_prompt.txt", "manual_aggregator_prompt.txt",
    "manual_main_submitter_config.json", "compiler_paper.txt", "compiler_outline.txt",
    "compiler_paper_saved.txt", "boost_state.json", "boost_api_log.txt", ".moto_backend.lock",
    ".moto_runtime.lock",
})
_RUNTIME_FILE_PATTERNS = ("compiler_last_10_*.txt", "summary_of_last_5_validator_rejections_for_submitter_*.txt")
# Private planning-note names, matched case-insensitively at any depth.
_PRIVATE_NOTE_NAMES = (
    "syntheticlib_*.md",
    "hardoj_*.md",
    "leanoj_*.md",
    "owner_deployment_*.md",
    "*_integration_plan*",
    "*_integration_remaining*.md",
    "handoff-manifest.json",
    "build_plan_*.md",
    "pending_tests",
)
_LOCAL_STATE_FILES = frozenset({
    ".moto_launcher_state.json",
    ".moto_last_instance.json",
    ".moto_update_notice.json",
    ".moto_installation_binding.json",
    "claude.local.md",
})
_DEV_LOG_NAMES = ("randomlog*.txt", "random log*.txt", "random dev log*.txt", "*.log")
_KEY_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".ppk", ".jks", ".keystore")
_CREDENTIAL_NAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc", "_netrc", ".git-credentials", ".pypirc",
})
_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp")
# Credential shapes with a fixed prefix; fake test values such as
# "sk-or-test-key" are too short to match.
_SECRET_PATTERNS = tuple((label, re.compile(pattern)) for label, pattern in (
    ("OpenRouter API key", r"sk-or-v1-[A-Za-z0-9]{20,}"),
    ("OpenAI-style API key", r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{40,}"),
    ("Anthropic API key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("xAI API key", r"xai-[A-Za-z0-9]{40,}"),
    ("GitHub token", r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    ("AWS access key", r"(?:AKIA|ASIA)(?!IOSFODNN7EXAMPLE)[0-9A-Z]{16}"),
    ("Google API key", r"AIza[0-9A-Za-z_-]{35}"),
    ("Slack token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("npm token", r"npm_[A-Za-z0-9]{36}"),
    ("Hugging Face token", r"hf_[A-Za-z0-9]{30,}"),
    ("GitLab token", r"glpat-[A-Za-z0-9_-]{20,}"),
    ("Stripe secret key", r"sk_live_[A-Za-z0-9]{20,}"),
    ("JSON web token", r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    # The second alternative is split so this source line cannot match itself.
    ("private key block", "-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----|PuTTY-User-Key-" "File"),
    ("Wolfram Alpha App ID", r"(?i:wolfram\w*|app[_-]?id)[^\n]{1,40}?(?=[A-Z0-9-]*\d)\b[A-Z0-9]{6}-[A-Z0-9]{10}\b"),
))
# A credential named by its key, on one line: "api_key = ...", "api_key: str = ...",
# os.environ["X_API_KEY"] = "...", setenv("X_PROXY_SECRET", "..."), "Bearer ...".
_TYPE_ANNOTATION = r"(?:[ \t]*:[ \t]*[A-Za-z_][\w.]{0,30}(?:\[[\w., ]{0,40}\])?(?=[ \t]*=))?"
_NAMED_SECRET = re.compile(
    r"(?i)(?:api[_-]?(?:key|token)|secret[_-]?(?:access[_-]?)?key|(?:client|proxy)[_-]?secret|"
    r"refresh[_-]?token|access[_-]?token|id[_-]?token|auth[_-]?token|password|bearer)"
    r"[\"'\]]*" + _TYPE_ANNOTATION + r"(?:[ \t]*[:=,][ \t]*[\"']?|[ \t]+)(?:bearer[ \t]+)?([A-Za-z0-9+/=~._-]{24,})"
)
_IDENTIFIER = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)+")
# Member paths such as settings.OPENROUTER_API_KEY_V2 or oauth2TokenResponse.accessToken.
_MEMBER_PATH = re.compile(r"[A-Za-z_$][A-Za-z_$]*(?:\d{1,2}[A-Za-z_$]*)?(?:\.[A-Za-z_$][A-Za-z_$]*(?:\d{1,2}[A-Za-z_$]*)?)+")
_FAKE_MARKER = re.compile(r"(?i)test|fake|dummy|example|sample|placeholder|mock|stub|invalid|redacted")
_HEX = re.compile(r"[0-9a-fA-F-]+")
_HOME_PATH = re.compile(
    r"(?i)(?:\b[a-z](?::|\$|%3a)(?:[\\/]+|%5c)users(?:[\\/]+|%5c)([a-z0-9][\w.-]*)"
    r"|(?<![\w.])/(?:mnt/|cygdrive/)?[a-z]/users/([a-z0-9][\w.-]*)"
    r"|(?<![\w.])/(?:users|home)/([a-z0-9][\w.-]*)"
    r"|\\\\wsl(?:\$|\.localhost)[\\/]+[^\\/\s]+[\\/]+home[\\/]+([a-z0-9][\w.-]*)"
    r"|\b[a-z]--users-([a-z0-9][a-z0-9_.]*))"
)
# Placeholder user names that example and test paths use.
_PLACEHOLDER_USERS = frozenset({
    "alice", "bob", "carol", "someone", "private", "user", "username", "me", "you", "name",
    "example", "runner", "runneradmin", "public", "default", "test", "dev", "moto",
    # System, service, and container accounts, and install-guide placeholders.
    "ubuntu", "ec2-user", "node", "vscode", "codespace", "gitpod", "linuxbrew", "appuser", "admin",
    "pi", "vagrant", "shared", "all", "yourname", "your_username", "yourusername", "your-username",
})
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PERSONAL_EMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "msn.com", "icloud.com", "me.com", "mac.com", "aol.com",
    "proton.me", "protonmail.com", "pm.me", "yandex.com", "yandex.ru", "mail.com", "mail.ru",
    "web.de", "fastmail.com", "hey.com", "tutanota.com", "tuta.io", "zoho.com", "qq.com", "163.com",
})
# Providers with regional domains (yahoo.co.uk, outlook.de, gmx.net, ...).
_PERSONAL_EMAIL_PROVIDERS = frozenset({"yahoo", "hotmail", "outlook", "live", "gmx"})
_ESCAPES = re.compile(r"\\(?:u[0-9a-fA-F]{4}|[nrt])")
_WORD = re.compile(r"[a-z0-9]+")
_ENDPOINT = re.compile(r"/v\d+/[a-z0-9_{}./-]*[a-z0-9_}]", re.IGNORECASE)
_PLACEHOLDER_SEGMENT = re.compile(r"\{[^/{}]*\}")
_HOST = re.compile(r"(?<![\w.-])(?:[a-z0-9-]+\.)+(?:com|io|dev|net|org|ai|app|cloud|internal)\b", re.IGNORECASE)
# Common values a plan may contain; they count only alongside copied plan text.
_WEAK_TOKEN_PATTERNS = (
    re.compile(r"\b(?:us|eu|ap|sa|ca|me|af)-(?:east|west|north|south|central|northeast|southeast)-\d\b"),
    re.compile(r"\b\d+\.\d+x\b"),
)


@dataclass(frozen=True)
class Violation:
    location: str
    reason: str
    detail: str


@dataclass
class PrivateText:
    """Ten-word windows and distinctive tokens from private planning files, each mapped to its source."""

    windows: dict[str, str] = field(default_factory=dict)
    strong_tokens: dict[str, str] = field(default_factory=dict)
    weak_tokens: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.windows or self.strong_tokens or self.weak_tokens)


@dataclass
class PushObjects:
    commits: list[str]
    tags: list[str]
    blobs: dict[str, str]
    loose_blobs: list[str]


class PublishSafetyError(RuntimeError):
    """The check could not run; callers fail closed."""


def private_path_reason(path: str) -> Optional[str]:
    """Return why a tracked path must never be published, or None when it may be."""
    if path in TRACKED_PLACEHOLDERS:
        return None
    lowered = path.replace("\\", "/").lower()
    segments = lowered.split("/")
    name = segments[-1]
    if any(segment in _PRIVATE_FOLDERS for segment in segments[:-1]):
        return "private planning or instance folder"
    if lowered.startswith("syntheticlib-trade/") and path not in PUBLISHABLE_TRADE_FILES:
        return "proof-trade material outside the published compatibility evidence"
    if any(fnmatchcase(name, pattern) for pattern in _PRIVATE_NOTE_NAMES):
        return "private planning note"
    if (
        lowered.startswith(("backend/data/", "backend/logs/"))
        or any(segment in _RUNTIME_FOLDERS for segment in segments[:-1])
        or name in _RUNTIME_FILES
        or any(fnmatchcase(name, pattern) for pattern in _RUNTIME_FILE_PATTERNS)
    ):
        return "runtime data from a MOTO data or log root"
    if name in _LOCAL_STATE_FILES or lowered == ".claude/settings.local.json":
        return "machine-local tool or installation state"
    if lowered.startswith(".claude/worktrees/"):
        return "local Claude Code worktree"
    if (
        name in {".env", ".envrc"} or name.startswith(".env.") or name.endswith(".env")
    ) and not name.endswith(_ENV_TEMPLATE_SUFFIXES):
        return "environment file"
    if name.endswith(_KEY_SUFFIXES) or name in _CREDENTIAL_NAMES:
        return "credential, key, or certificate file"
    if any(fnmatchcase(name, pattern) for pattern in _DEV_LOG_NAMES):
        return "development log"
    return None


def _looks_random(value: str) -> bool:
    """A credential value mixes letters and digits with high entropy; code and test values do not."""
    if (
        _IDENTIFIER.fullmatch(value)
        or _MEMBER_PATH.fullmatch(value)
        or _FAKE_MARKER.search(value)
        or not (re.search(r"\d", value) and re.search(r"[A-Za-z]", value))
    ):
        return False
    counts = Counter(value)
    entropy = -sum(count / len(value) * math.log2(count / len(value)) for count in counts.values())
    # Hex has 16 symbols, so random hex values sit lower than base62 or base64 ones.
    return entropy >= (3.0 if _HEX.fullmatch(value) else 3.5)


def personal_email(address: str) -> bool:
    domain = address.rsplit("@", 1)[-1].strip().lower() if "@" in address else ""
    return domain in _PERSONAL_EMAIL_DOMAINS or domain.split(".", 1)[0] in _PERSONAL_EMAIL_PROVIDERS


def content_reasons(text: str, public_emails: frozenset[str] = frozenset()) -> list[str]:
    """Return why a line or message must not be published."""
    reasons = [label for label, pattern in _SECRET_PATTERNS if pattern.search(text)]
    if any(_looks_random(match.group(1)) for match in _NAMED_SECRET.finditer(text)):
        reasons.append("credential value")
    for match in _HOME_PATH.finditer(text):
        user = next(group for group in match.groups() if group)
        if user.lower() not in _PLACEHOLDER_USERS:
            reasons.append("user home directory path")
            break
    if any(
        personal_email(match.group(0)) and match.group(0).lower() not in public_emails
        for match in _EMAIL.finditer(text)
    ):
        reasons.append("personal email address")
    return reasons


def _message_reasons(message: str, public_emails: frozenset[str]) -> list[str]:
    """Content reasons for a commit or tag message, checked line by line like file content."""
    reasons: list[str] = []
    for line in message.splitlines():
        reasons.extend(reason for reason in content_reasons(line, public_emails) if reason not in reasons)
    return reasons


def _iter_windows(text: str) -> Iterable[str]:
    words = _WORD.findall(_ESCAPES.sub(" ", text).lower())
    for index in range(len(words) - WINDOW_WORDS + 1):
        yield " ".join(words[index:index + WINDOW_WORDS])


def _endpoint_tokens(text: str) -> set[str]:
    tokens = set()
    for match in _ENDPOINT.finditer(text):
        segments = _PLACEHOLDER_SEGMENT.sub("{}", match.group(0).lower()).split("/")
        # "/v1/x/{}/y" also yields "/v1/x" and "/v1/x/{}", so code that builds the
        # route from pieces still matches.
        for end in range(3, len(segments) + 1):
            tokens.add("/".join(segments[:end]))
    return tokens


def _strong_tokens(text: str) -> set[str]:
    return _endpoint_tokens(text) | {match.group(0).lower() for match in _HOST.finditer(text)}


def _weak_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for pattern in _WEAK_TOKEN_PATTERNS for match in pattern.finditer(text)}


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, 9)
        except OSError:
            process.kill()


def _git_raw(
    args: Sequence[str],
    repo: Path,
    stdin: Optional[bytes] = None,
    timeout: int = GIT_TIMEOUT_SECONDS,
    ok_codes: Sequence[int] = (0,),
) -> bytes:
    try:
        process = subprocess.Popen(
            ["git", *args], cwd=repo, env=_GIT_ENV,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name != "nt",
        )
    except OSError as exc:
        raise PublishSafetyError(f"git {' '.join(args[:3])} could not run: {exc}") from exc
    try:
        out, err = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Kill helpers too (git-remote-https), or Windows waits on their pipes.
        _kill_tree(process)
        process.communicate()
        raise PublishSafetyError(f"git {' '.join(args[:3])} timed out after {timeout} s") from exc
    if process.returncode not in ok_codes:
        raise PublishSafetyError(f"git {' '.join(args[:3])} failed: {err.decode('utf-8', 'replace').strip()}")
    return out


def _git(args: Sequence[str], repo: Path, stdin: Optional[str] = None, timeout: int = GIT_TIMEOUT_SECONDS) -> str:
    raw = _git_raw(args, repo, stdin.encode("utf-8") if stdin is not None else None, timeout)
    return raw.decode("utf-8", "replace")


def _object_types(shas: Iterable[str], repo: Path) -> dict[str, str]:
    shas = sorted(set(shas))
    if not shas:
        return {}
    output = _git(["cat-file", "--batch-check=%(objectname) %(objecttype)"], repo, "\n".join(shas) + "\n")
    types = {}
    for line in output.splitlines():
        sha, _, kind = line.partition(" ")
        if kind and kind != "missing":
            types[sha] = kind
    return types


def published_shas(target: str, repo: Path, fetch_from: Optional[str] = None) -> tuple[set[str], set[str]]:
    """Return (branch/tag tips, pull-request tips) the target advertises, kept to local objects.

    Branch tips this clone has not fetched fail closed: without them, history the
    remote already publishes cannot be told apart from new commits. The advice
    names every missing branch, since a single-branch clone's plain fetch skips them.
    """
    listing = _git(["ls-remote", target], repo, timeout=LS_REMOTE_TIMEOUT_SECONDS)
    heads: dict[str, str] = {}
    tags: set[str] = set()
    pulls: set[str] = set()
    for line in listing.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            heads[ref] = sha
        elif ref.startswith("refs/tags/"):
            tags.add(sha)
        elif ref.startswith("refs/pull/"):
            pulls.add(sha)
    present = _object_types([*heads.values(), *tags, *pulls], repo)
    missing = sorted(ref for ref, sha in heads.items() if sha not in present)
    if missing:
        raise PublishSafetyError(
            f"{fetch_from or target} has {len(missing)} branch tip(s) this clone has not fetched; "
            f"run `git fetch {fetch_from or target} {' '.join(missing)}` and push again"
        )
    published = {sha for sha in [*heads.values(), *tags] if sha in present}
    return published, {sha for sha in pulls if sha in present}


def objects_to_publish(tips: Iterable[str], published: Iterable[str], repo: Path) -> PushObjects:
    """Every object reachable from the tips but not from what the remote already has."""
    tips = [tip for tip in tips if tip and tip != ZERO_SHA]
    objects = PushObjects([], [], {}, [])
    if not tips:
        return objects
    request = "\n".join([*tips, *(f"^{sha}" for sha in sorted(published))]) + "\n"
    listing = [line for line in _git(["rev-list", "--objects", "--stdin"], repo, request).splitlines() if line]
    types = _object_types((line.split(" ", 1)[0] for line in listing), repo)
    for line in listing:
        sha, _, path = line.partition(" ")
        kind = types.get(sha)
        if kind == "commit":
            objects.commits.append(sha)
        elif kind == "tag":
            objects.tags.append(sha)
        elif kind == "blob":
            if path:
                objects.blobs.setdefault(sha, path)
            else:
                objects.loose_blobs.append(sha)
    return objects


def _ignored_paths(paths: Iterable[str], worktree: Path, excludes_file: Optional[Path]) -> set[str]:
    paths = sorted(set(paths))
    if not paths:
        return set()
    excludes = str(excludes_file) if excludes_file else ""
    raw = _git_raw(
        ["-c", f"core.excludesFile={excludes}", "check-ignore", "--no-index", "--stdin", "-z"],
        worktree, ("\0".join(paths) + "\0").encode("utf-8"), ok_codes=(0, 1),
    )
    return {path for path in raw.decode("utf-8", "replace").split("\0") if path}


def _read_blobs(shas: Sequence[str], repo: Path) -> dict[str, bytes]:
    if not shas:
        return {}
    raw = _git_raw(["cat-file", "--batch"], repo, ("\n".join(shas) + "\n").encode("ascii"))
    contents: dict[str, bytes] = {}
    offset = 0
    for sha in shas:
        header_end = raw.index(b"\n", offset)
        header = raw[offset:header_end].split()
        if len(header) < 3:
            raise PublishSafetyError(f"git object {sha} could not be read")
        size = int(header[2])
        start = header_end + 1
        contents[sha] = raw[start:start + size]
        offset = start + size + 1
    return contents


def _decoded_text(content: bytes) -> Optional[str]:
    """Readable text of a blob, or None when it is binary."""
    if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return content.decode("utf-32", "replace")
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16", "replace")
    if b"\0" in content[:8192]:
        return None
    return content.decode("utf-8", "replace")


def _published_blobs(published: Iterable[str], repo: Path) -> list[str]:
    types = _object_types(published, repo)
    blobs = {sha for sha, kind in types.items() if kind == "blob"}
    # Annotated tags are skipped: ls-remote also lists the object each one peels to.
    for sha, kind in types.items():
        if kind not in {"commit", "tree"}:
            continue
        for entry in _git(["ls-tree", "-r", "-z", f"{sha}^{{tree}}"], repo).split("\0"):
            meta = entry.partition("\t")[0].split()
            if len(meta) == 3 and meta[1] == "blob":
                blobs.add(meta[2])
    return sorted(blobs)


_FINGERPRINT_REASONS = frozenset({
    "private planning note",
    "private planning or instance folder",
    "proof-trade material outside the published compatibility evidence",
})
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "test-results", "playwright-report",
    ".pytest_cache", ".moto_instances",
})


def _private_sources(root: Path) -> list[tuple[Path, str]]:
    """Private planning files anywhere in a working tree, tracked or not."""
    sources = []
    for directory, folders, files in os.walk(root):
        relative_dir = Path(directory).relative_to(root).as_posix()
        prefix = "" if relative_dir == "." else f"{relative_dir}/"
        folders[:] = [
            folder for folder in folders
            if folder not in _SKIP_DIRS and f"{prefix}{folder}" not in {"backend/data", "backend/logs"}
        ]
        for name in files:
            relative = f"{prefix}{name}"
            # Copies of the published compatibility evidence are public text.
            if private_path_reason(relative) in _FINGERPRINT_REASONS and "compatibility" not in relative.lower().split("/")[:-1]:
                sources.append((Path(directory) / name, relative))
    return sources


def _is_partial_clone(repo: Path) -> bool:
    raw = _git_raw(
        ["config", "--get-regexp", r"^(extensions\.partialclone|remote\..*\.promisor)$"], repo, ok_codes=(0, 1)
    ).decode("utf-8", "replace")
    return any(line.split(" ", 1)[-1].strip().lower() not in {"", "false"} for line in raw.splitlines())


def load_private_text(roots: Iterable[Path], repo: Path, published: Iterable[str]) -> PrivateText:
    """Fingerprint the private planning files present in this clone's working trees.

    Windows and tokens that the remote already publishes (including the released
    compatibility evidence) are dropped, so only still-private text is matched.
    """
    sources = [source for root in {root.resolve() for root in roots if root.is_dir()} for source in _private_sources(root)]
    private = PrivateText()
    for source, label in sources:
        try:
            if source.stat().st_size > MAX_PRIVATE_SOURCE_BYTES:
                continue
            text = _decoded_text(source.read_bytes())
        except OSError:
            continue
        if not text:
            continue
        for window in _iter_windows(text):
            private.windows.setdefault(window, label)
        for token in _strong_tokens(text):
            private.strong_tokens.setdefault(token, label)
        for token in _weak_tokens(text):
            private.weak_tokens.setdefault(token, label)
    if not private:
        return private
    if _is_partial_clone(repo):
        # Reading every published file would fetch each missing one separately.
        raise PublishSafetyError(
            "this is a partial (blobless) clone, so the published files needed to tell private plan "
            "text from public text are missing; run `git fetch --refetch origin` or use a full clone"
        )
    for content in _read_blobs(_published_blobs(published, repo), repo).values():
        if len(content) > MAX_PRIVATE_SOURCE_BYTES:
            continue
        text = _decoded_text(content)
        if not text:
            continue
        for window in [window for window in _iter_windows(text) if window in private.windows]:
            private.windows.pop(window, None)
        for token in _strong_tokens(text):
            private.strong_tokens.pop(token, None)
        for token in _weak_tokens(text):
            private.weak_tokens.pop(token, None)
    return private


def _private_text_reasons(text: str, private: PrivateText) -> list[tuple[str, str]]:
    """(reason, detail) pairs naming the local plan file and the matched words or token."""
    reasons: list[tuple[str, str]] = []
    if private.windows:
        run = 0
        for window in _iter_windows(text):
            run = run + 1 if window in private.windows else 0
            if run >= MIN_WINDOW_RUN:
                reasons.append((
                    "text copied from a private planning file",
                    f"\"{' '.join(window.split()[:6])} ...\" ({private.windows[window]})",
                ))
                break
    for token in sorted(_strong_tokens(text) & private.strong_tokens.keys()) if private.strong_tokens else ():
        reasons.append(("an endpoint or host from a private planning file", f"{token} ({private.strong_tokens[token]})"))
    if reasons and private.weak_tokens:
        for token in sorted(_weak_tokens(text) & private.weak_tokens.keys()):
            reasons.append(("a region or rate from a private planning file", f"{token} ({private.weak_tokens[token]})"))
    return reasons


def _declared_submodules(commit: str, repo: Path, cache: dict[str, set[str]]) -> set[str]:
    try:
        sha = _git(["rev-parse", f"{commit}:.gitmodules"], repo).strip()
    except PublishSafetyError:
        return set()
    if sha not in cache:
        try:
            listing = _git(["config", "--blob", sha, "--get-regexp", r"^submodule\..*\.path$"], repo)
        except PublishSafetyError:
            listing = ""
        cache[sha] = {line.split(" ", 1)[1] for line in listing.splitlines() if " " in line}
    return cache[sha]


def check_push(
    objects: PushObjects,
    repo: Path,
    worktree: Optional[Path] = None,
    public_commits: Iterable[str] = (),
    private: Optional[PrivateText] = None,
    excludes_file: Optional[Path] = None,
    public_emails: frozenset[str] = frozenset(),
) -> list[Violation]:
    worktree = worktree or repo
    private = private or PrivateText()
    public_commits = set(public_commits)
    violations: list[Violation] = []

    def add(location: str, reason: str, detail: str) -> None:
        violation = Violation(location, reason, detail)
        if violation not in violations:
            violations.append(violation)

    # Paths: every path a new commit adds or changes, even when its content is
    # already public (an empty file, a copy, a rename), plus every new blob's path.
    blob_paths: dict[str, set[str]] = {sha: {path} for sha, path in objects.blobs.items()}
    pushed_paths: set[str] = set(objects.blobs.values())
    submodules: dict[str, set[str]] = {}
    for commit in objects.commits:
        changed = _git(
            ["diff-tree", "-r", "-z", "--root", "-c", "--no-commit-id", "--name-only", "--diff-filter=ACMRT", commit],
            repo,
        )
        pushed_paths.update(path for path in changed.split("\0") if path)
        for entry in _git(["ls-tree", "-r", "-z", "--full-tree", commit], repo).split("\0"):
            if not entry:
                continue
            meta, _, path = entry.partition("\t")
            mode, kind, sha = meta.split()
            if kind == "commit" and path not in _declared_submodules(commit, repo, submodules):
                add(commit[:10], "undeclared embedded repository", path)
            if sha in blob_paths:
                blob_paths[sha].add(path)
    for paths_ in blob_paths.values():
        pushed_paths.update(paths_)
    ignored = _ignored_paths(pushed_paths, worktree, excludes_file)
    for path in sorted(pushed_paths):
        reason = private_path_reason(path) or ("ignored by .gitignore" if path in ignored else None)
        if reason:
            add("file", reason, path)

    contents = _read_blobs(sorted(blob_paths) + objects.loose_blobs, repo)
    for sha in objects.loose_blobs:
        add("ref", "ref points directly at a blob", sha[:10])
    for sha, paths_ in blob_paths.items():
        text = _decoded_text(contents.get(sha, b""))
        path = sorted(paths_)[0]
        if text is None:
            if not path.lower().endswith(_IMAGE_SUFFIXES):
                add("file", "binary file the check cannot read", path)
            continue
        flagged: set[str] = set()
        for number, line in enumerate(text.splitlines(), 1):
            for reason in content_reasons(line, public_emails):
                if reason not in flagged:
                    flagged.add(reason)
                    add("file", reason, f"{path}:{number}")
        for reason, detail in _private_text_reasons(text, private):
            add("file", reason, f"{path}: {detail}")

    for commit in objects.commits:
        if commit in public_commits:
            continue
        header = _git(["show", "-s", "--format=%ae%x00%ce%x00%B", commit], repo)
        author, committer, message = (header.split("\0", 2) + ["", ""])[:3]
        short = commit[:10]
        for role, address in (("author", author.strip()), ("committer", committer.strip())):
            if personal_email(address) and address.lower() not in public_emails:
                add(short, f"personal {role} email address", address)
        for reason in _message_reasons(message, public_emails):
            add(short, f"{reason} in the commit message", "(message)")
        for reason, detail in _private_text_reasons(message, private):
            add(short, f"{reason} in the commit message", detail)

    for tag in objects.tags:
        headers, _, message = _git(["cat-file", "tag", tag], repo).partition("\n\n")
        tagger = next((line for line in headers.splitlines() if line.startswith("tagger ")), "")
        address = tagger.partition("<")[2].partition(">")[0]
        if personal_email(address) and address.lower() not in public_emails:
            add(tag[:10], "personal tagger email address", address)
        for reason in _message_reasons(message, public_emails):
            add(tag[:10], f"{reason} in the tag message", "(message)")
        for reason, detail in _private_text_reasons(message, private):
            add(tag[:10], f"{reason} in the tag message", detail)
    return violations


def parse_pre_push_lines(stdin: Iterable[str]) -> list[str]:
    return [line for line in stdin if line.strip()]


def parse_pre_push_tips(lines: Iterable[str]) -> list[str]:
    """Local SHAs from git's pre-push input: '<local ref> <local sha> <remote ref> <remote sha>'."""
    tips = []
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[1] != ZERO_SHA:
            tips.append(parts[1])
    return tips


def _toplevel(cwd: Path) -> Path:
    return Path(_git(["rev-parse", "--show-toplevel"], cwd).strip())


def _common_dir(toplevel: Path) -> Path:
    common = Path(_git(["rev-parse", "--git-common-dir"], toplevel).strip())
    return (common if common.is_absolute() else toplevel / common).resolve()


def _config_values(repo: Path, key: str) -> list[str]:
    try:
        values = _git_raw(["config", "--get-all", key], repo, ok_codes=(0, 1))
    except PublishSafetyError:
        return []
    return values.decode("utf-8", "replace").split()


def private_remotes(repo: Path) -> set[str]:
    return set(_config_values(repo, "moto.publishSafety.privateRemotes"))


def public_reference_remotes(repo: Path, exclude: str) -> list[str]:
    """Remotes explicitly configured as public; a remote's name alone never makes it public."""
    return [remote for remote in _config_values(repo, "moto.publishSafety.publicRemotes") if remote != exclude]


def run_check(
    tips: Sequence[str],
    target: str,
    cwd: Path,
    excludes_file: Optional[Path] = None,
    reference_remotes: Sequence[str] = (),
    fetch_from: Optional[str] = None,
) -> tuple[list[Violation], int]:
    toplevel = _toplevel(cwd)
    published, pulls = published_shas(target, toplevel, fetch_from)
    for remote in reference_remotes:
        more, more_pulls = published_shas(remote, toplevel, remote)
        published |= more
        pulls |= more_pulls
    objects = objects_to_publish(tips, published, toplevel)
    public_commits: set[str] = set()
    if pulls:
        request = "\n".join([*pulls, *(f"^{sha}" for sha in published)]) + "\n"
        public_commits = set(_git(["rev-list", "--stdin"], toplevel, request).split())
    public_emails: frozenset[str] = frozenset()
    if published or pulls:
        listing = _git(["log", "--format=%ae%n%ce", "--stdin"], toplevel, "\n".join([*published, *pulls]) + "\n")
        public_emails = frozenset(line.strip().lower() for line in listing.splitlines() if line.strip())
    private = load_private_text([toplevel, _common_dir(toplevel).parent], toplevel, published)
    violations = check_push(objects, toplevel, toplevel, public_commits, private, excludes_file, public_emails)
    return violations, len(objects.commits)


def format_violations(violations: Sequence[Violation]) -> str:
    lines = ["Publish-safety check FAILED: this push would publish private content:"]
    for violation in violations:
        lines.append(f"  {violation.location}  {violation.reason}: {violation.detail}")
    lines += [
        "",
        "GitHub keeps every pushed commit downloadable by SHA, even after a force-push,",
        "so remove the content from these commits before pushing (for example with an",
        "interactive rebase or git filter-repo) instead of deleting it in a later commit.",
    ]
    if any("private planning file" in violation.reason for violation in violations):
        lines.append(
            "If a matched plan detail is now meant to be public, reword or remove it in the named "
            "local plan file, then push again."
        )
    if any("email" in violation.reason for violation in violations):
        lines.append(
            "For a personal email, commit with your GitHub noreply address (git config user.email) and "
            "rewrite the commits, for example: git rebase --exec \"git commit --amend --no-edit --reset-author\" <base>. "
            "If the commits are already public upstream and you push to a fork, run "
            "`git config --add moto.publishSafety.publicRemotes upstream` instead."
        )
    return "\n".join(lines)


_HOOK_SCRIPT = f"""#!/bin/sh
# {HOOK_MARKER}: installed by `python -m scripts.check_publish_safety --install`.
# Refuses pushes that would publish private plans, runtime data, or credentials.
# A pre-push hook that was here before runs first, with the same input.
hook_dir=$(cd "$(dirname "$0")" && pwd)
input=$(cat)
if [ -f "$hook_dir/{CHAINED_HOOK_NAME}" ]; then
  printf '%s\\n' "$input" | "$hook_dir/{CHAINED_HOOK_NAME}" "$@" || exit $?
fi
# Git passes no refs when the push uploads nothing (up to date, or rejected).
[ -n "$input" ] || exit 0
cd "$(git rev-parse --show-toplevel)" || exit 1
for python in python3 python py; do
  # Skip launcher stubs (such as the Windows Store alias) that cannot run Python 3.10+.
  if "$python" -c "import sys; sys.exit(sys.version_info < (3, 10))" >/dev/null 2>&1; then
    printf '%s\\n' "$input" | "$python" "$hook_dir/{INSTALLED_NAME}" --pre-push "$@"
    exit $?
  fi
done
echo "pre-push: Python 3.10+ not found; refusing to push without the publish-safety check." >&2
exit 1
"""


def hooks_dir(cwd: Path) -> Path:
    return _common_dir(_toplevel(cwd)) / "hooks"


def _effective_hooks_dir(cwd: Path) -> Path:
    toplevel = _toplevel(cwd)
    hooks = Path(_git(["rev-parse", "--git-path", "hooks"], toplevel).strip())
    return (hooks if hooks.is_absolute() else toplevel / hooks).resolve()


def _is_guard_hook(hook: Path) -> bool:
    return hook.exists() and HOOK_MARKER in hook.read_text(encoding="utf-8", errors="replace")


def install(cwd: Path) -> str:
    """Install the hook into the shared hooks directory; return a status line."""
    toplevel = _toplevel(cwd)
    target = hooks_dir(cwd)
    effective = _effective_hooks_dir(cwd)
    if effective != target:
        raise PublishSafetyError(
            f"git runs hooks from {effective} (core.hooksPath), not {target}. Unset core.hooksPath, "
            f"or install there by hand: copy this file to that folder as {INSTALLED_NAME} and run it "
            "with --pre-push from that folder's pre-push hook, passing git's input through."
        )
    target.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    if source != (target / INSTALLED_NAME).resolve():
        shutil.copyfile(source, target / INSTALLED_NAME)
    if (toplevel / ".gitignore").is_file():
        shutil.copyfile(toplevel / ".gitignore", target / GITIGNORE_SNAPSHOT_NAME)
    hook = target / "pre-push"
    chained = target / CHAINED_HOOK_NAME
    message = f"Publish-safety pre-push hook installed in {target}."
    if hook.exists() and not _is_guard_hook(hook):
        if chained.exists() and chained.read_bytes() != hook.read_bytes():
            raise PublishSafetyError(
                f"{hook} was replaced by another tool while {chained} still holds an earlier hook; "
                f"merge them into {chained}, delete {hook}, and run --install again."
            )
        hook.replace(chained)
        message += f" The existing pre-push hook now runs first from {chained}."
    hook.write_bytes(_HOOK_SCRIPT.encode("utf-8"))
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return message


def hook_installed(cwd: Path) -> bool:
    try:
        target = hooks_dir(cwd)
        effective = _effective_hooks_dir(cwd)
        toplevel = _toplevel(cwd)
    except PublishSafetyError:
        return False
    installed = target / INSTALLED_NAME
    snapshot = target / GITIGNORE_SNAPSHOT_NAME
    gitignore = toplevel / ".gitignore"
    return (
        effective == target
        and _is_guard_hook(target / "pre-push")
        and (target / "pre-push").read_bytes() == _HOOK_SCRIPT.encode("utf-8")
        and installed.exists()
        and installed.read_bytes() == Path(__file__).resolve().read_bytes()
        and (not gitignore.is_file() or (snapshot.exists() and snapshot.read_bytes() == gitignore.read_bytes()))
    )


def main(argv: Optional[Sequence[str]] = None, stdin: TextIO = sys.stdin) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pre-push", nargs="*", metavar="REMOTE",
        help="Run as git's pre-push hook: the remote name and push URL, with pushed refs on stdin.",
    )
    parser.add_argument("--install", action="store_true", help="Install or refresh the pre-push hook, then check HEAD.")
    parser.add_argument("--remote", default="origin", help="Remote to compare with outside hook mode (default: origin).")
    parser.add_argument("refs", nargs="*", help="Refs to check outside hook mode (default: HEAD).")
    args = parser.parse_args(argv)
    cwd = Path.cwd()

    try:
        excludes_file: Optional[Path] = None
        if args.install:
            print(install(cwd))
        toplevel = _toplevel(cwd)
        if args.pre_push is not None:
            name = args.pre_push[0] if args.pre_push else "origin"
            target = args.pre_push[1] if len(args.pre_push) > 1 else name
            if name in private_remotes(toplevel):
                print(f"publish-safety: {name} is listed in moto.publishSafety.privateRemotes; not checked.")
                return 0
            lines = parse_pre_push_lines(stdin)
            if not lines:
                raise PublishSafetyError(
                    "git passed no refs to the hook; another hook may have read them first. "
                    "Reinstall with --install so that hook is chained correctly"
                )
            snapshot = Path(__file__).resolve().with_name(GITIGNORE_SNAPSHOT_NAME)
            excludes_file = snapshot if snapshot.is_file() else None
            tips = parse_pre_push_tips(lines)
        else:
            name = args.remote
            target = _git(["remote", "get-url", "--push", name], cwd).strip()
            tips = [_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd).strip() for ref in (args.refs or ["HEAD"])]
        violations, checked = run_check(
            tips, target, cwd, excludes_file, public_reference_remotes(toplevel, exclude=name), fetch_from=name,
        )
    except Exception as exc:  # Any failure refuses the push rather than letting it through.
        print(f"Publish-safety check could not run, so the push is refused: {exc!r}", file=sys.stderr)
        return 2

    if violations:
        print(format_violations(violations), file=sys.stderr)
        return 1
    print(f"Publish-safety check passed ({checked} unpublished commit(s) checked against {name}).")
    if args.pre_push is None and not args.install and not hook_installed(cwd):
        print(
            "WARNING: the publish-safety pre-push hook is not installed (or is outdated) in this clone, "
            "so a plain `git push` is not checked. Install it with: python -m scripts.check_publish_safety --install"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
