import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import pytest

from scripts import check_publish_safety as guard
from scripts.check_publish_safety import (
    PUBLISHABLE_TRADE_FILES,
    TRACKED_PLACEHOLDERS,
    content_reasons,
    main,
    parse_pre_push_tips,
    private_path_reason,
)
from scripts.run_github_checks import SYNTHETICLIB_EVIDENCE, build_steps


REPO_ROOT = Path(__file__).resolve().parents[2]
# Sample values are assembled at runtime so this file never contains a literal
# the guard itself would refuse to publish.
USER = "ma" + "x"
REGION = "us-east-" + "2"
RATE = "1." + "5x"
OTHER_RATE = "1." + "2x"
PRIVATE_PATHS = (
    "PLANS/example-plan.md",
    "plans/notes.json",
    "docs/PLANS/copied.md",
    "docs/plans/copied.md",
    "syntheticlib-trade/contracts/example.json",
    "syntheticlib-trade/compatibility/example.json",
    "SYNTHETICLIB_EXAMPLE_NOTES.md",
    "docs/SYNTHETICLIB_EXAMPLE_NOTES.md",
    "HARDOJ_EXAMPLE.md",
    "LEANOJ_EXAMPLE.md",
    "OWNER_DEPLOYMENT_EXAMPLE.md",
    "EXAMPLE_INTEGRATION_PLAN.md",
    "handoff-manifest.json",
    "PENDING_TESTS",
    "backend/data/compiler_paper_saved.txt",
    "backend/data/chroma_db.quarantine-20261028T000000Z/chroma.sqlite3",
    "backend/data/syntheticlib_trade/proof_trade_ledger.json",
    "backend/data/proofs/proof_1.json",
    "backend/logs/backend.txt",
    "moto_data/auto_sessions/run_1/session_metadata.json",
    "moto_data/runtime_settings.json",
    "moto_data/auto_translations/translation_store.sqlite3",
    "elsewhere/manual_compiler_prompt.txt",
    "tests/fixtures/user_uploads/client_notes.txt",
    "debug_copy/compiler_last_10_rejections.txt",
    "debug_copy/Summary_Of_Last_5_Validator_Rejections_For_Submitter_1.txt",
    ".moto_instances/abc/data/runtime_settings.json",
    "backend/data/.moto_installation_binding.json",
    ".moto_launcher_state.json",
    ".claude/settings.local.json",
    ".claude/worktrees/agent-1/README.md",
    "CLAUDE.local.md",
    ".env",
    ".envrc",
    "deploy/.env.production",
    "docker/hosted.env",
    "certs/server.pem",
    "keys/deploy.ppk",
    ".git-credentials",
    "random dev log.txt",
    "randomlog2.txt",
    "backend.log",
)
PUBLISHABLE_PATHS = (
    "README.md",
    "backend/data/.gitkeep",
    "backend/data/auto_final_answer/.gitkeep",
    "backend/data/proofs/.gitkeep",
    "backend/logs/.gitkeep",
    "backend/shared/runtime_settings.py",
    "backend/shared/syntheticlib_trade_compatibility.py",
    "backend/shared/syntheticlib_source_selection.py",
    "backend/leanoj/prompts.py",
    "syntheticlib-trade/compatibility/README.md",
    "syntheticlib-trade/compatibility/compatibility-matrix.json",
    ".claude/rules/api-key-controls.md",
    "docs/explanations.md",
    ".env.example",
    "docker/hosted.env.example",
    "tests/fixtures/syntheticlib4/release_manifest.json",
)


@pytest.mark.parametrize("path", PRIVATE_PATHS)
def test_private_paths_are_flagged(path: str) -> None:
    assert private_path_reason(path)


@pytest.mark.parametrize("path", PUBLISHABLE_PATHS)
def test_publishable_paths_are_allowed(path: str) -> None:
    assert private_path_reason(path) is None


def _git_checkout() -> bool:
    return shutil.which("git") is not None and (REPO_ROOT / ".git").exists()


def _publishable_files() -> list[str]:
    """Everything `git add -A` would commit: tracked files plus unignored new ones."""
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return sorted({path for path in output.split("\0") if path and (REPO_ROOT / path).is_file()})


def test_self_contained_lists_match_the_repository() -> None:
    # The installed hook runs outside the checkout, so it keeps its own copies.
    assert PUBLISHABLE_TRADE_FILES == {*SYNTHETICLIB_EVIDENCE, "syntheticlib-trade/compatibility/README.md"}
    if _git_checkout():
        placeholders = {path for path in _publishable_files() if path.startswith(("backend/data/", "backend/logs/"))}
        assert placeholders == TRACKED_PLACEHOLDERS


def test_everything_committable_passes_the_guard_rules() -> None:
    # A rule that fires on the repository itself would block every push of it,
    # including the guard's own source and these tests. Where this clone holds
    # private plans, their text is checked too, against what origin last showed.
    if not _git_checkout():
        pytest.skip("needs a git checkout")
    tips = subprocess.run(
        ["git", "for-each-ref", "--format=%(objectname)", "refs/remotes/origin/"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    private = guard.load_private_text([REPO_ROOT], REPO_ROOT, tips)
    hits = []
    for path in _publishable_files():
        if private_path_reason(path):
            hits.append((path, "private path"))
            continue
        text = guard._decoded_text((REPO_ROOT / path).read_bytes())
        if text is None:
            if not path.lower().endswith(guard._IMAGE_SUFFIXES):
                hits.append((path, "binary"))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            hits.extend((f"{path}:{number}", reason) for reason in content_reasons(line))
        hits.extend((path, reason) for reason, _ in guard._private_text_reasons(text, private))
    assert hits == []


def test_gitignore_ignores_every_private_path_kind() -> None:
    # Checked through git itself, so a re-including negation in any form fails here.
    if not _git_checkout():
        pytest.skip("needs a git checkout")

    def ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "-c", "core.excludesFile=", "check-ignore", "-q", "--no-index", path],
            cwd=REPO_ROOT, capture_output=True, check=False,
        )
        return result.returncode == 0

    # The guard alone covers lowercase variants on case-sensitive file systems and
    # data-root files outside the default roots.
    guard_only = {
        "docs/plans/copied.md", "plans/notes.json", "moto_data/auto_sessions/run_1/session_metadata.json",
        "moto_data/runtime_settings.json", "moto_data/auto_translations/translation_store.sqlite3",
        "elsewhere/manual_compiler_prompt.txt",
        "tests/fixtures/user_uploads/client_notes.txt", "debug_copy/compiler_last_10_rejections.txt",
        "debug_copy/Summary_Of_Last_5_Validator_Rejections_For_Submitter_1.txt",
    }
    assert [path for path in PRIVATE_PATHS if path not in guard_only and not ignored(path)] == []
    assert [path for path in PUBLISHABLE_PATHS if not path.endswith(".example") and ignored(path)] == []


def test_content_check_flags_credentials() -> None:
    fake = {
        "GitHub token": "ghp_" + "A1b2" * 9,
        "OpenAI-style API key": "sk-" + "Ab3" * 16,
        "JSON web token": "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 12,
        "npm token": "npm_" + "x" * 36,
        "AWS access key": "AKIA" + "ABCD1234EFGH5678",
        "private key block": "-----BEGIN PGP PRIVATE KEY" + " BLOCK-----",
    }
    for label, token in fake.items():
        assert content_reasons(f"value = {token}") == [label]
    secret = "7f3c9a1e" + "Kq2Lm8Np4Rs6Tu0Vw"
    assert content_reasons("PuTTY-User-Key-" + "File-3: ssh-ed25519") == ["private key block"]
    assert content_reasons("WOLFRAM_ALPHA_APP_ID=" + "6PJ2RY-" + "T7V8QQ9KLL") == ["Wolfram Alpha App ID"]
    for line in (
        '{"refresh_token": "' + "v1.Mj" + "Ay9xQ7pZ3kL8mN2rT5wB" + '"}',
        "Authorization: Bearer " + secret,
        'os.environ["MOTO_SAKANA_FUGU_API_KEY"] = "' + secret + '"',
        'monkeypatch.setenv("MOTO_KIMI_CODE_API_KEY", "' + secret + '")',
        'keyring.set_password("MOTO", "openrouter_api_key", "' + secret + '")',
        "MOTO_INTERNAL_PROXY_SECRET=" + secret,
        "VITE_MOTO_DESKTOP_API_TOKEN=" + secret,
        'desktop_api_token: Optional[str] = "' + secret + '"',
        'const apiKey: string = "' + secret + '"',
        "MOTO_INTERNAL_PROXY_SECRET=" + "4f3a2b1c" + "0d9e8f7a6b5c4d3e2f1a0b9c",
    ):
        assert content_reasons(line) == ["credential value"], line
    assert content_reasons('wolfram_alpha_api_key: Optional[str] = "' + "6PJ2RY-" + 'T7V8QQ9KLL"') == [
        "Wolfram Alpha App ID"
    ]
    for line in (
        'monkeypatch.setenv("MOTO_KIMI_CODE_API_KEY", "test-kimi-code-key-12345")',
        'api_key = "fake-openrouter-key-for-unit-tests-1"',
        "api_key = settings.OPENROUTER_API_KEY_V2",
        "const accessToken = oauth2TokenResponse.accessToken;",
    ):
        assert content_reasons(line) == [], line
    assert content_reasons("key = sk-or-" + "test-streaming-key") == []
    assert content_reasons("aws = AKIA" + "IOSFODNN7EXAMPLE") == []
    assert content_reasons("api_key = self._resolve_api_key_from_environment") == []
    assert content_reasons("access_token = token_response_v2_payload_value") == []


def test_content_check_flags_real_home_paths_and_personal_emails() -> None:
    backslash = "\\"
    for path in (
        "C:" + f"{backslash}Users{backslash}{USER}{backslash}Desktop",
        f"/c/Users/{USER}/moto",
        f"/mnt/c/Users/{USER}/moto",
        f"/cygdrive/c/Users/{USER}",
        f"/home/{USER}/moto",
        f"/Users/{USER}/Library",
        f"~/.claude/projects/C--Users-{USER}-Desktop-MOTO/memory",
        "file:///c%3A/Users/" + USER + "/x",
        "C:%5CUsers%5C" + USER + "%5Cx",
        backslash * 2 + "wsl$" + backslash + "Ubuntu" + backslash + "home" + backslash + USER,
        backslash * 2 + "wsl.localhost/Ubuntu/home/" + USER,
    ):
        assert content_reasons(path) == ["user home directory path"], path
    assert content_reasons("C:" + f"{backslash}Users{backslash}alice{backslash}Desktop and /home/someone/x") == []
    assert content_reasons(f"GET https://example.com/users/{USER} and /api/users/{USER}") == []
    assert content_reasons("cd /home/ubuntu/MOTO && WORKDIR /home/node/app") == []
    assert content_reasons("Mail me at someone@" + "yahoo.co.uk or x@" + "pm.me") == ["personal email address"]
    address = "pat.personal@" + "gmail.com"
    assert content_reasons(f"Co-authored-by: Pat <{address}>") == ["personal email address"]
    assert content_reasons(f"Co-authored-by: Pat <{address}>", frozenset({address})) == []
    assert content_reasons("Contact security@intrafere.com") == []


def test_content_check_stays_fast_on_huge_lines() -> None:
    started = time.perf_counter()
    content_reasons("a1b2c3d4" * 125_000)
    content_reasons("password" + " " * 200_000)
    guard._strong_tokens("a." * 200_000)
    assert time.perf_counter() - started < 5


# --- end-to-end checks in throwaway repositories ---------------------------------

def _run(args, cwd, **kwargs):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, **kwargs)


def _identity(email="dev@example.com"):
    return ["-c", "user.name=Dev", "-c", f"user.email={email}", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]


def _bare(path: Path) -> Path:
    _run(["init", "-q", "--bare", "-b", "main", str(path)], path.parent)
    return path


@pytest.fixture
def remote_pair(tmp_path: Path):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    remote = _bare(tmp_path / "remote.git")
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["init", "-q", "-b", "main"], repo)
    _run(["remote", "add", "origin", remote.as_posix()], repo)
    return repo, remote


def _commit(repo: Path, files: dict, message: str = "change", email: str = "dev@example.com") -> str:
    for relative, content in files.items():
        path = repo / relative
        if content is None:
            _run(["rm", "-q", relative], repo)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        _run(["add", "-f", relative], repo)
    _run([*_identity(email), "commit", "-q", "-m", message], repo)
    return _run(["rev-parse", "HEAD"], repo).stdout.strip()


def _publish(repo: Path, ref: str = "main") -> None:
    _run(["push", "-q", "--no-verify", "origin", f"HEAD:refs/heads/{ref}"], repo)


def _write_hook(path: Path, script: str, executable: bool = True) -> None:
    # Hook installers such as `git lfs install` write executable hooks; git skips the rest.
    path.write_text(script, encoding="utf-8", newline="\n")
    mode = path.stat().st_mode
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    path.chmod(mode | exec_bits if executable else mode & ~exec_bits)


def _push(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "push", "origin", "HEAD:refs/heads/main"], cwd=repo, capture_output=True, text=True)


def _check(repo: Path, tips=("HEAD",), target=None, references=()):
    shas = [_run(["rev-parse", tip], repo).stdout.strip() for tip in tips]
    target = target or _run(["remote", "get-url", "--push", "origin"], repo).stdout.strip()
    violations, _ = guard.run_check(shas, target, repo, reference_remotes=references)
    return [(v.reason, v.detail) for v in violations]


def _details(found, reason):
    return [detail for found_reason, detail in found if found_reason == reason]


def test_unpublished_commits_are_checked_even_when_a_later_commit_removes_the_file(remote_pair) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    _commit(repo, {"PLANS/plan.md": "private\n"})
    _commit(repo, {"PLANS/plan.md": None})
    assert ("private planning or instance folder", "PLANS/plan.md") in _check(repo)


def test_private_names_are_flagged_even_when_their_content_is_already_public(remote_pair) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n", "backend/data/.gitkeep": ""})
    _publish(repo)
    _commit(repo, {".env": "", "HARDOJ_EXAMPLE.md": "public\n", "moto_data/runtime_settings.json": "{}\n"})
    found = _check(repo)
    assert ("environment file", ".env") in found
    assert ("private planning note", "HARDOJ_EXAMPLE.md") in found
    assert ("runtime data from a MOTO data or log root", "moto_data/runtime_settings.json") in found


def test_stale_or_foreign_tracking_refs_do_not_hide_a_leak(remote_pair, tmp_path: Path) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    leaked = _commit(repo, {"PLANS/plan.md": "private\n"})
    _commit(repo, {"PLANS/plan.md": None})
    _run(["update-ref", "refs/remotes/origin/old", leaked], repo)
    laptop = _bare(tmp_path / "laptop.git")
    _run(["remote", "add", "laptop", laptop.as_posix()], repo)
    _run(["push", "-q", "--no-verify", "laptop", f"{leaked}:refs/heads/main"], repo)
    _run(["fetch", "-q", "laptop"], repo)
    assert ("private planning or instance folder", "PLANS/plan.md") in _check(repo)


def test_the_push_url_is_checked_not_the_fetch_url(remote_pair, tmp_path: Path, monkeypatch, capsys) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    leaked = _commit(repo, {"PLANS/plan.md": "private\n"})
    tip = _commit(repo, {"PLANS/plan.md": None})
    _run(["push", "-q", "--no-verify", "origin", f"{leaked}:refs/heads/backup", f"{tip}:refs/heads/main"], repo)
    public = _bare(tmp_path / "public.git")
    _run(["remote", "set-url", "--push", "origin", public.as_posix()], repo)
    monkeypatch.chdir(repo)
    push = io.StringIO(f"refs/heads/main {tip} refs/heads/main {'0' * 40}\n")
    assert main(["--pre-push", "origin", public.as_posix()], stdin=push) == 1
    assert "PLANS/plan.md" in capsys.readouterr().err


def test_an_unreachable_push_target_or_missing_input_fails_closed(remote_pair, tmp_path: Path, monkeypatch) -> None:
    repo, remote = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    with pytest.raises(guard.PublishSafetyError):
        _check(repo, target=(tmp_path / "missing.git").as_posix())
    monkeypatch.chdir(repo)
    assert main(["--pre-push", "origin", remote.as_posix()], stdin=io.StringIO("\n")) == 2


def test_unfetched_remote_branches_fail_closed_with_a_working_fetch_command(remote_pair, tmp_path: Path) -> None:
    repo, remote = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    other = tmp_path / "other"
    _run(["clone", "-q", remote.as_posix(), str(other)], tmp_path)
    _commit(other, {"NEWS.md": "from elsewhere\n"})
    _run(["push", "-q", "--no-verify", "origin", "HEAD:refs/heads/feature"], other)
    with pytest.raises(guard.PublishSafetyError, match="refs/heads/feature"):
        _check(repo)
    _run(["fetch", "-q", remote.as_posix(), "refs/heads/feature"], repo)
    assert _check(repo) == []


def test_replace_refs_do_not_hide_what_the_push_uploads(remote_pair) -> None:
    repo, _ = remote_pair
    base = _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    leaked = _commit(repo, {"PLANS/plan.md": "private\n"})
    _run(["checkout", "-q", "-b", "clean", base], repo)
    clean = _commit(repo, {"notes.txt": "clean\n"})
    _run(["checkout", "-q", "main"], repo)
    _run(["replace", leaked, clean], repo)
    assert ("private planning or instance folder", "PLANS/plan.md") in _check(repo, tips=(leaked,))


def test_already_published_content_does_not_block_later_pushes(remote_pair) -> None:
    repo, _ = remote_pair
    _commit(repo, {"tests/fixtures/server.pem": "fixture\n", "a.txt": "a\n"})
    _publish(repo)
    _commit(repo, {"a.txt": "b\n"})
    assert _check(repo) == []


def test_binary_utf16_tags_and_gitlinks_are_checked(remote_pair) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    key = "sk-or-v1-" + "a" * 32
    _commit(repo, {
        "docs/archive.zip": b"PK\x03\x04\x00\x00binary",
        "docs/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00",
        "notes/unicode.txt": ("﻿key=" + key).encode("utf-16"),
        ".gitmodules": '[submodule "libfoo"]\n\tpath = vendor/libfoo\n\turl = https://example.com/libfoo\n',
    })
    blob = _run(["hash-object", "-w", "--stdin"], repo, input="loose\n").stdout.strip()
    _run(["update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},vendor/lib"], repo)
    _run(["update-index", "--add", "--cacheinfo", f"160000,{'2' * 40},vendor/libfoo"], repo)
    _run([*_identity(), "commit", "-q", "-m", "gitlink"], repo)
    _run([*_identity(), "tag", "-a", "v1", "-m", f"release token {key}"], repo)
    _run(["tag", "loose", blob], repo)
    found = _check(repo, tips=("HEAD", "v1", "loose"))
    assert ("binary file the check cannot read", "docs/archive.zip") in found
    assert not any(detail == "docs/logo.png" for _, detail in found)
    assert ("OpenRouter API key", "notes/unicode.txt:1") in found
    assert ("undeclared embedded repository", "vendor/lib") in found
    assert ("undeclared embedded repository", "vendor/libfoo") not in found
    assert ("OpenRouter API key in the tag message", "(message)") in found
    assert any(reason == "ref points directly at a blob" for reason, _ in found)


SHARED = "the user facing disclosure text that is already public in the code base for every reader today"
SECRET = (
    "the multiplier table grants special credit to each accepted proof in the private launch "
    "window before the partner review closes and the second phase opens"
)


def _write_plan(repo: Path, *lines: str) -> None:
    (repo / "PLANS").mkdir(exist_ok=True)
    (repo / "PLANS" / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_private_plan_text_and_tokens_are_caught_but_published_text_is_not(remote_pair) -> None:
    repo, _ = remote_pair
    _write_plan(repo, SHARED, SECRET, "Call /v1/example-trades/{trade_id}/settle on staging.example-internal.dev")
    # Published text repeating a plan phrase must not break the subtraction.
    _commit(repo, {"src/panel.txt": SHARED + "\n", "src/panel_copy.txt": f"{SHARED}\n{SHARED}\n"})
    _publish(repo)
    _commit(repo, {
        "docs/notes.md": SECRET + "\n",
        "docs/fixture.json": json.dumps({"text": SECRET.replace(" proof ", " proof\n")}) + "\n",
        "src/client.py": 'url = "/v1/example-trades/" + trade_id + "/settle"\n',
        "src/panel.txt": SHARED + " again\n",
    })
    found = _check(repo)
    copied = _details(found, "text copied from a private planning file")
    assert any(detail.startswith("docs/notes.md: ") and "PLANS/plan.md" in detail for detail in copied)
    assert any(detail.startswith("docs/fixture.json: ") for detail in copied)
    assert "src/client.py: /v1/example-trades (PLANS/plan.md)" in _details(
        found, "an endpoint or host from a private planning file"
    )
    assert not any(detail.startswith("src/panel.txt") for _, detail in found)


def test_common_regions_and_rates_count_only_alongside_copied_text(remote_pair) -> None:
    repo, _ = remote_pair
    _write_plan(repo, SECRET, f"Rates rise {RATE} and {OTHER_RATE} in {REGION} for partners")
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    twelve_words = " ".join(SECRET.split()[:12])
    _commit(repo, {
        "docs/a.md": twelve_words + "\n",
        "src/zoom.js": f'export const ZOOM = ["1.0x", "{RATE}", "{OTHER_RATE}"];\n',
        "docs/deploy.md": f"Deploy to {REGION} at {RATE} the base rate\n",
    })
    assert _check(repo) == []
    _commit(repo, {"docs/leak.md": f"{SECRET} in {REGION}\n"})
    found = _check(repo)
    assert any(detail.startswith("docs/leak.md") for detail in _details(found, "a region or rate from a private planning file"))


def test_personal_emails_are_flagged_unless_already_public(remote_pair) -> None:
    repo, _ = remote_pair
    base = _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    contributor = _commit(repo, {"a.txt": "pr\n"}, email="someone@" + "gmail.com")
    _run(["push", "-q", "--no-verify", "origin", f"{contributor}:refs/pull/1/head"], repo)
    assert _check(repo) == []
    # A cherry-pick rewrites the SHA but publishes no new address.
    _run(["checkout", "-q", "-b", "picked", base], repo)
    _run([*_identity("someone@" + "gmail.com"), "cherry-pick", contributor], repo)
    assert _check(repo) == []
    _commit(repo, {"b.txt": "mine\n"}, email="me@" + "outlook.com")
    _commit(repo, {"c.txt": "trailer\n"}, message="Fix\n\nCo-authored-by: Pat <pat.personal@" + "gmail.com>")
    _commit(repo, {"d.txt": "redact\n"}, message="Never log the Kimi Code api_key\n\nbackend/shared/lean4_client.py now redacts it.")
    assert sorted(reason for reason, _ in _check(repo)) == [
        "personal author email address",
        "personal committer email address",
        "personal email address in the commit message",
    ]


def test_an_upstream_reference_remote_counts_as_published_for_forks(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    upstream = _bare(tmp_path / "upstream.git")
    fork = _bare(tmp_path / "fork.git")
    seed = tmp_path / "seed"
    _run(["clone", "-q", upstream.as_posix(), str(seed)], tmp_path)
    _commit(seed, {"README.md": "public\n"})
    _run(["push", "-q", "--no-verify", "origin", "HEAD:refs/heads/main"], seed)
    _run(["push", "-q", "--no-verify", fork.as_posix(), "HEAD:refs/heads/main"], seed)
    _commit(seed, {"web.md": "edited on the web\n"}, email="maintainer@" + "gmail.com")
    _run(["push", "-q", "--no-verify", "origin", "HEAD:refs/heads/main"], seed)
    contributor = tmp_path / "contributor"
    _run(["clone", "-q", fork.as_posix(), str(contributor)], tmp_path)
    _run(["remote", "add", "upstream", upstream.as_posix()], contributor)
    _run(["fetch", "-q", "upstream"], contributor)
    _run(["checkout", "-q", "-b", "fix", "upstream/main"], contributor)
    _commit(contributor, {"fix.md": "fix\n"})
    assert any("email" in reason for reason, _ in _check(contributor))
    # A remote's name alone never makes it public; the fork opts in explicitly.
    assert guard.public_reference_remotes(contributor, exclude="origin") == []
    _run(["config", "--add", "moto.publishSafety.publicRemotes", "upstream"], contributor)
    assert guard.public_reference_remotes(contributor, exclude="origin") == ["upstream"]
    assert _check(contributor, references=["upstream"]) == []


def test_remote_tags_on_blobs_do_not_break_the_check(remote_pair) -> None:
    repo, _ = remote_pair
    _write_plan(repo, SECRET)
    _commit(repo, {"README.md": "public\n"})
    blob = _run(["hash-object", "-w", "--stdin"], repo, input="maintainer key\n").stdout.strip()
    _run(["tag", "maintainer-key", blob], repo)
    _run([*_identity(), "tag", "-a", "maintainer-key-annotated", "-m", "key", blob], repo)
    _run(["push", "-q", "--no-verify", "origin", "HEAD:refs/heads/main", "--tags"], repo)
    _commit(repo, {"a.txt": "a\n"})
    assert _check(repo) == []


def test_pre_push_mode_reads_git_input_and_honors_every_private_remote(remote_pair, monkeypatch, capsys) -> None:
    repo, remote = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    tip = _commit(repo, {".env": "OPENROUTER_API_KEY=x\n"})
    monkeypatch.chdir(repo)
    push = io.StringIO(f"refs/heads/main {tip} refs/heads/main {'0' * 40}\n")
    assert main(["--pre-push", "origin", remote.as_posix()], stdin=push) == 1
    assert "environment file: .env" in capsys.readouterr().err
    _run(["config", "--add", "moto.publishSafety.privateRemotes", "origin"], repo)
    _run(["config", "--add", "moto.publishSafety.privateRemotes", "laptop"], repo)
    assert main(["--pre-push", "origin", remote.as_posix()], stdin=io.StringIO("")) == 0


def test_pre_push_input_skips_deleted_refs() -> None:
    zero = "0" * 40
    lines = [f"refs/heads/feature {'a' * 40} refs/heads/feature {zero}\n", f"(delete) {zero} refs/heads/old {'b' * 40}\n"]
    assert parse_pre_push_tips(lines) == ["a" * 40]


def test_install_writes_a_shared_hook_and_gitignore_snapshot_from_any_directory(remote_pair, monkeypatch) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n", ".gitignore": "secrets/\n", "sub/x.txt": "x\n"})
    monkeypatch.chdir(repo / "sub")
    assert "installed" in guard.install(repo / "sub")
    hooks = repo / ".git" / "hooks"
    assert (hooks / guard.INSTALLED_NAME).read_bytes() == Path(guard.__file__).read_bytes()
    assert (hooks / guard.GITIGNORE_SNAPSHOT_NAME).read_text(encoding="utf-8") == "secrets/\n"
    assert guard.HOOK_MARKER in (hooks / "pre-push").read_text(encoding="utf-8")
    assert guard.hook_installed(repo / "sub")
    (repo / ".gitignore").write_text("secrets/\nmore/\n", encoding="utf-8")
    assert not guard.hook_installed(repo)


def test_an_existing_hook_is_chained_and_both_see_the_pushed_refs(remote_pair) -> None:
    if shutil.which("sh") is None:
        pytest.skip("needs a POSIX shell to run git hooks")
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    # Like `git lfs pre-push`, this hook reads all of git's input.
    _write_hook(hooks / "pre-push", '#!/bin/sh\ncat > "$(dirname "$0")/seen.txt"\nexit 0\n')
    assert "runs first" in guard.install(repo)
    assert guard.hook_installed(repo)
    _commit(repo, {"PLANS/plan.md": "private\n"})
    push = _push(repo)
    assert push.returncode != 0
    assert "PLANS/plan.md" in push.stderr
    assert "refs/heads/main" in (hooks / "seen.txt").read_text(encoding="utf-8")
    # Pushes that upload nothing get empty input from git and must not be refused.
    _run(["reset", "-q", "--hard", "HEAD~1"], repo)
    (hooks / "seen.txt").unlink()
    up_to_date = _push(repo)
    assert up_to_date.returncode == 0, up_to_date.stderr
    # If git runs the hooks here, the earlier one sees git's empty input, not a stray blank line.
    seen = hooks / "seen.txt"
    assert not seen.exists() or seen.read_text(encoding="utf-8") == ""
    # A tool re-adding the same hook (git lfs update --force) does not block a reinstall.
    _write_hook(hooks / "pre-push", (hooks / guard.CHAINED_HOOK_NAME).read_text(encoding="utf-8"))
    guard.install(repo)
    assert guard.hook_installed(repo)


def test_a_non_executable_existing_hook_stays_skipped_like_git_skips_it(remote_pair) -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("needs POSIX file modes and a POSIX shell to run git hooks")
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    # Git never ran this hook, so the guard must not start running it (or fail on it).
    _write_hook(hooks / "pre-push", '#!/bin/sh\ntouch "$(dirname "$0")/ran.txt"\nexit 1\n', executable=False)
    assert "not executable" in guard.install(repo)
    _commit(repo, {"PLANS/plan.md": "private\n"})
    refused = _push(repo)
    assert refused.returncode != 0
    assert "PLANS/plan.md" in refused.stderr
    _run(["reset", "-q", "--hard", "HEAD~1"], repo)
    _commit(repo, {"docs/public.md": "public\n"})
    clean = _push(repo)
    assert clean.returncode == 0, clean.stderr
    assert not (hooks / "ran.txt").exists()


def test_install_refuses_any_hooks_path_without_changing_it(remote_pair, tmp_path: Path, monkeypatch) -> None:
    repo, _ = remote_pair
    _commit(repo, {"README.md": "public\n"})
    _run(["config", "core.hooksPath", ".githooks"], repo)
    with pytest.raises(guard.PublishSafetyError, match="core.hooksPath"):
        guard.install(repo)
    assert _run(["config", "--get", "core.hooksPath"], repo).stdout.strip() == ".githooks"
    _run(["config", "--unset", "core.hooksPath"], repo)
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text(f"[core]\n\thooksPath = {(tmp_path / 'scanner-hooks').as_posix()}\n", encoding="utf-8")
    monkeypatch.setitem(guard._GIT_ENV, "GIT_CONFIG_GLOBAL", str(global_config))
    with pytest.raises(guard.PublishSafetyError, match="core.hooksPath"):
        guard.install(repo)
    assert not guard.hook_installed(repo)


def test_github_parity_installs_and_runs_the_publish_check_first() -> None:
    first = build_steps(python="python", npm="npm")[0]
    assert first.slug == "publish-safety"
    assert first.command == ("python", "-m", "scripts.check_publish_safety", "--install")


def test_private_notes_anywhere_in_the_tree_are_fingerprinted(remote_pair) -> None:
    repo, _ = remote_pair
    (repo / "docs").mkdir()
    (repo / "docs" / "LEANOJ_EXAMPLE_NOTES.md").write_text(SECRET + "\n", encoding="utf-8")
    (repo / "syntheticlib-trade" / "contracts").mkdir(parents=True)
    (repo / "syntheticlib-trade" / "contracts" / "draft.md").write_text(
        "POST /v1/example-settlements/{batch_id}\n", encoding="utf-8"
    )
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    _commit(repo, {"docs/public.md": SECRET + "\n", "src/api.py": 'ROUTE = "/v1/example-settlements/"\n'})
    found = _check(repo)
    assert any(detail.startswith("docs/public.md") for detail in _details(found, "text copied from a private planning file"))
    assert any(detail.startswith("src/api.py") for detail in _details(found, "an endpoint or host from a private planning file"))


def test_partial_clones_with_private_plans_fail_fast(remote_pair) -> None:
    repo, _ = remote_pair
    _write_plan(repo, SECRET)
    _commit(repo, {"README.md": "public\n"})
    _publish(repo)
    _run(["config", "remote.origin.promisor", "true"], repo)
    with pytest.raises(guard.PublishSafetyError, match="partial"):
        _check(repo)
