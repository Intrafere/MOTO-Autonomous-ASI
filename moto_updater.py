"""
Build 1/2 updater helpers for the MOTO launcher.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_JSON_PATH = REPO_ROOT / "package.json"
LOCAL_MANIFEST_PATH = REPO_ROOT / "moto-update-manifest.json"
LAUNCHER_STATE_PATH = REPO_ROOT / ".moto_launcher_state.json"
LAUNCHER_LAST_INSTANCE_PATH = REPO_ROOT / ".moto_last_instance.json"
LAUNCHER_ENTRYPOINT_ENV = "MOTO_LAUNCHER_ENTRYPOINT"

_DEFAULT_MANIFEST = {
    "manifest_version": 1,
    "version": "0.0.0-dev",
    "build_commit": "dev",
    "update_channel": "main",
    "api_contract_version": "build5-v1",
}

_DEFAULT_PRESERVED_ROOTS = {
    ".git",
    ".moto_instances",
    "backend/data",
    "backend/logs",
    "frontend/node_modules",
    "frontend/dist",
    ".venv",
    "venv",
    "env",
    "ENV",
}
_DEFAULT_PRESERVED_FILES = {
    ".env",
    ".env.local",
    ".moto_launcher_state.json",
    ".moto_last_instance.json",
    "frontend/.env",
    "frontend/.env.local",
}
_WINDOWS_YES = 6


@dataclass(frozen=True)
class BuildManifest:
    version: str
    build_commit: str
    update_channel: str
    api_contract_version: str
    manifest_version: int = 1

    @property
    def short_commit(self) -> str:
        return self.build_commit[:7] if self.build_commit else "unknown"


@dataclass(frozen=True)
class InstallState:
    kind: str
    label: str
    can_auto_apply: bool
    reason: str
    active_instance_count: int = 0
    git_branch: str | None = None
    git_upstream: str | None = None
    git_remote_url: str | None = None


@dataclass(frozen=True)
class UpdateCheckResult:
    local_manifest: BuildManifest
    remote_manifest: BuildManifest | None
    install_state: InstallState
    error: str | None = None
    warning: str | None = None
    metadata_source: str = "none"

    @property
    def update_available(self) -> bool:
        if self.remote_manifest is None:
            return False
        return self.remote_manifest.build_commit != self.local_manifest.build_commit

    @property
    def can_apply_update(self) -> bool:
        return (
            self.update_available
            and self.remote_manifest is not None
            and self.metadata_source == "manifest"
            and self.install_state.can_auto_apply
        )


@dataclass
class _CopyJournal:
    created_files: list[str]
    overwritten_files: list[str]
    overwritten_directories: list[str]

    def __init__(self) -> None:
        self.created_files = []
        self.overwritten_files = []
        self.overwritten_directories = []


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_package_json() -> dict:
    payload = _read_json(PACKAGE_JSON_PATH)
    return payload if isinstance(payload, dict) else {}


def _coerce_manifest(payload: dict | None) -> BuildManifest:
    data = dict(_DEFAULT_MANIFEST)
    package_json = _load_package_json()
    version = str(package_json.get("version", "")).strip()
    if version:
        data["version"] = version

    if isinstance(payload, dict):
        for key in ("version", "build_commit", "update_channel", "api_contract_version"):
            value = str(payload.get(key, "")).strip()
            if value:
                data[key] = value
        try:
            data["manifest_version"] = int(payload.get("manifest_version", data["manifest_version"]))
        except (TypeError, ValueError):
            data["manifest_version"] = int(_DEFAULT_MANIFEST["manifest_version"])

    return BuildManifest(
        version=str(data["version"]),
        build_commit=str(data["build_commit"]),
        update_channel=str(data["update_channel"]),
        api_contract_version=str(data["api_contract_version"]),
        manifest_version=int(data["manifest_version"]),
    )


def load_local_manifest() -> BuildManifest:
    return _coerce_manifest(_read_json(LOCAL_MANIFEST_PATH))


def _normalize_repo_slug(url: str) -> str | None:
    raw = (url or "").strip()
    if not raw:
        return None

    cleaned = raw.rstrip("/")
    for prefix in ("git+https://", "https://", "http://", "ssh://git@"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    if cleaned.startswith("git@github.com:"):
        cleaned = cleaned[len("git@github.com:") :]
    elif cleaned.startswith("github.com/"):
        cleaned = cleaned[len("github.com/") :]

    cleaned = cleaned.removesuffix(".git")
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return None


def _sanitize_instance_id(raw: str | None) -> str | None:
    normalized = (raw or "").strip()
    if not normalized:
        return None
    cleaned = []
    for character in normalized:
        if character.isalnum() or character in "._-":
            cleaned.append(character)
        else:
            cleaned.append("_")
    collapsed = "".join(cleaned).strip("_")
    return collapsed or None


def _official_repo_slug() -> str:
    package_json = _load_package_json()
    repository = package_json.get("repository", {})
    if isinstance(repository, dict):
        repo_url = str(repository.get("url", "")).strip()
    else:
        repo_url = str(repository or "").strip()

    repo_slug = _normalize_repo_slug(repo_url)
    if not repo_slug:
        raise RuntimeError("Could not determine the official GitHub repository URL from package.json.")
    return repo_slug


def _manifest_url_for_channel(update_channel: str) -> str:
    repo_slug = _official_repo_slug()
    channel = (update_channel or "main").strip() or "main"
    return f"https://raw.githubusercontent.com/{repo_slug}/{channel}/moto-update-manifest.json"


def _package_json_url_for_channel(update_channel: str) -> str:
    repo_slug = _official_repo_slug()
    channel = (update_channel or "main").strip() or "main"
    return f"https://raw.githubusercontent.com/{repo_slug}/{channel}/package.json"


def _branch_api_url_for_channel(update_channel: str) -> str:
    repo_slug = _official_repo_slug()
    channel = (update_channel or "main").strip() or "main"
    return f"https://api.github.com/repos/{repo_slug}/branches/{channel}"


def archive_url_for_manifest(manifest: BuildManifest) -> str:
    repo_slug = _official_repo_slug()
    return f"https://github.com/{repo_slug}/archive/{manifest.build_commit}.zip"


def _fetch_json_url(url: str, timeout_seconds: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MOTO-Build1-Updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload from {url}")
    return payload


def fetch_remote_manifest(local_manifest: BuildManifest, timeout_seconds: int = 10) -> BuildManifest:
    payload = _fetch_json_url(_manifest_url_for_channel(local_manifest.update_channel), timeout_seconds)
    return _coerce_manifest(payload)


def fetch_branch_head_fallback(local_manifest: BuildManifest, timeout_seconds: int = 10) -> BuildManifest:
    package_payload = _fetch_json_url(_package_json_url_for_channel(local_manifest.update_channel), timeout_seconds)
    branch_payload = _fetch_json_url(_branch_api_url_for_channel(local_manifest.update_channel), timeout_seconds)

    version = str(package_payload.get("version", "")).strip() or local_manifest.version
    commit = str(branch_payload.get("commit", {}).get("sha", "")).strip()
    if not commit:
        raise RuntimeError("GitHub branch metadata did not include a branch-head commit SHA.")

    return BuildManifest(
        version=version,
        build_commit=commit,
        update_channel=local_manifest.update_channel,
        api_contract_version=local_manifest.api_contract_version,
        manifest_version=local_manifest.manifest_version,
    )


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            return


def consume_internal_launcher_args(argv: list[str]) -> tuple[list[str], list[Path]]:
    passthrough: list[str] = []
    cleanup_paths: list[Path] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--moto-cleanup-update":
            if index + 1 < len(argv):
                cleanup_paths.append(Path(argv[index + 1]))
                index += 2
                continue
        passthrough.append(arg)
        index += 1
    return passthrough, cleanup_paths


def _is_pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def _load_launcher_state() -> dict:
    payload = _read_json(LAUNCHER_STATE_PATH)
    if not isinstance(payload, dict):
        return {"instances": []}
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        instances = []
    return {"instances": instances}


def _save_launcher_state(payload: dict) -> None:
    if not payload.get("instances"):
        cleanup_path(LAUNCHER_STATE_PATH)
        return
    _write_json(LAUNCHER_STATE_PATH, payload)


def cleanup_launcher_state() -> list[dict]:
    payload = _load_launcher_state()
    active_instances: list[dict] = []
    for instance in payload.get("instances", []):
        if not isinstance(instance, dict):
            continue
        backend_pid = _coerce_int(instance.get("backend_window_pid"))
        frontend_pid = _coerce_int(instance.get("frontend_window_pid"))
        if _is_pid_running(backend_pid) or _is_pid_running(frontend_pid):
            active_instances.append(instance)

    _save_launcher_state({"instances": active_instances})
    return active_instances


def register_active_instance(
    *,
    instance_id: str,
    backend_window_pid: int,
    frontend_window_pid: int,
    backend_port: int,
    frontend_port: int,
    data_root: str,
    log_root: str,
    secret_namespace: str | None,
    storage_prefix: str | None,
) -> None:
    active_instances = cleanup_launcher_state()
    active_instances.append(
        {
            "instance_id": instance_id,
            "backend_window_pid": backend_window_pid,
            "frontend_window_pid": frontend_window_pid,
            "backend_port": backend_port,
            "frontend_port": frontend_port,
            "data_root": data_root,
            "log_root": log_root,
            "secret_namespace": secret_namespace,
            "storage_prefix": storage_prefix,
        }
    )
    _save_launcher_state({"instances": active_instances})


def load_last_instance_record() -> dict | None:
    """Return the most recently launched non-default instance record, or None.

    Used to preserve a stable secret_namespace / data_root / storage_prefix across
    relaunches when the default ports are temporarily busy. Without this the
    launcher would mint a fresh timestamped instance_id on every relaunch, which
    changes the OS-keyring service name and makes the saved OpenRouter / Wolfram
    Alpha API keys look "missing" on startup.
    """
    payload = _read_json(LAUNCHER_LAST_INSTANCE_PATH)
    if not isinstance(payload, dict):
        return None
    instance_id = payload.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None
    return payload


def save_last_instance_record(
    *,
    instance_id: str,
    data_root: str,
    log_root: str,
    secret_namespace: str | None,
    storage_prefix: str | None,
) -> None:
    """Persist the last launched non-default instance so it can be reused on relaunch."""
    _write_json(
        LAUNCHER_LAST_INSTANCE_PATH,
        {
            "instance_id": instance_id,
            "data_root": data_root,
            "log_root": log_root,
            "secret_namespace": secret_namespace,
            "storage_prefix": storage_prefix,
        },
    )


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _git_output(args: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git_checkout_matches_repo() -> bool:
    code, output, _ = _git_output(["rev-parse", "--show-toplevel"])
    if code != 0 or not output:
        return False
    try:
        return Path(output).resolve() == REPO_ROOT.resolve()
    except OSError:
        return False


def classify_install_state(active_instances: list[dict]) -> InstallState:
    active_count = len(active_instances)
    active_reason = ""
    if active_count:
        plural = "s are" if active_count != 1 else " is"
        active_reason = (
            f"{active_count} MOTO instance{plural} still running from this install. "
            "Close both launcher-managed MOTO services before applying an update."
        )

    has_git_metadata = (REPO_ROOT / ".git").exists()
    if not _git_checkout_matches_repo():
        if has_git_metadata:
            return InstallState(
                kind="dirty_git_checkout",
                label="Git checkout",
                can_auto_apply=False,
                reason=active_reason or "This install has git metadata, but the checkout could not be inspected safely for automatic updates.",
                active_instance_count=active_count,
            )
        reason = active_reason or "ZIP / extracted consumer install."
        return InstallState(
            kind="zip_install",
            label="ZIP / extracted consumer install",
            can_auto_apply=active_count == 0,
            reason=reason,
            active_instance_count=active_count,
        )

    code, branch, _ = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        return InstallState(
            kind="dirty_git_checkout",
            label="Git checkout",
            can_auto_apply=False,
            reason=active_reason or "Could not determine the current git branch for this checkout.",
            active_instance_count=active_count,
        )

    _, upstream, _ = _git_output(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    _, status_output, _ = _git_output(["status", "--porcelain", "--untracked-files=no"])
    _, remote_url, _ = _git_output(["remote", "get-url", "origin"])
    official_slug = _official_repo_slug()
    remote_matches = _normalize_repo_slug(remote_url) == official_slug
    dirty_checkout = bool(status_output.strip())
    clean_checkout = branch == "main" and upstream == "origin/main" and remote_matches and not dirty_checkout
    if clean_checkout:
        return InstallState(
            kind="clean_git_clone",
            label="Clean git clone on main",
            can_auto_apply=active_count == 0,
            reason=active_reason or "Clean git checkout tracking origin/main.",
            active_instance_count=active_count,
            git_branch=branch,
            git_upstream=upstream,
            git_remote_url=remote_url,
        )

    reason_parts = []
    if active_reason:
        reason_parts.append(active_reason)
    if branch != "main":
        reason_parts.append(f"Current branch is '{branch}', not 'main'.")
    if upstream != "origin/main":
        reason_parts.append("This checkout does not track origin/main.")
    if remote_url and not remote_matches:
        reason_parts.append("This checkout's origin remote does not match the official MOTO repository.")
    if dirty_checkout:
        reason_parts.append("Tracked files have local modifications.")
    if not reason_parts:
        reason_parts.append("This git checkout is not a safe fast-forward target for automatic updates.")

    return InstallState(
        kind="dirty_git_checkout",
        label="Dirty or non-standard git checkout",
        can_auto_apply=False,
        reason=" ".join(reason_parts),
        active_instance_count=active_count,
        git_branch=branch,
        git_upstream=upstream or None,
        git_remote_url=remote_url or None,
    )


def check_for_updates() -> UpdateCheckResult:
    local_manifest = load_local_manifest()
    active_instances = cleanup_launcher_state()
    install_state = classify_install_state(active_instances)
    try:
        remote_manifest = fetch_remote_manifest(local_manifest)
        return UpdateCheckResult(
            local_manifest,
            remote_manifest,
            install_state,
            metadata_source="manifest",
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            try:
                fallback_manifest = fetch_branch_head_fallback(local_manifest)
                return UpdateCheckResult(
                    local_manifest,
                    fallback_manifest,
                    install_state,
                    warning=(
                        "GitHub main is reachable, but `moto-update-manifest.json` is not published there yet. "
                        "The launcher can compare branch-head builds, but automatic update-apply stays disabled until the manifest exists on main."
                    ),
                    metadata_source="branch_head_fallback",
                )
            except (RuntimeError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as fallback_exc:
                return UpdateCheckResult(
                    local_manifest,
                    None,
                    install_state,
                    error=(
                        f"{exc}. The fallback branch-head lookup also failed: {fallback_exc}"
                    ),
                )
        return UpdateCheckResult(local_manifest, None, install_state, error=str(exc))
    except (RuntimeError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return UpdateCheckResult(local_manifest, None, install_state, error=str(exc))


def show_yes_no_dialog(title: str, message: str) -> bool:
    if sys.platform == "win32":
        try:
            import ctypes

            response = ctypes.windll.user32.MessageBoxW(None, message, title, 0x00000004 | 0x00000040)
            return response == _WINDOWS_YES
        except Exception:
            pass

    print()
    print(message)
    choice = input("Apply update now? [y/N]: ").strip().lower()
    return choice in {"y", "yes"}


def show_info_dialog(title: str, message: str) -> None:
    print()
    print(f"{title}: {message}")


UPDATE_NOTICE_PATH = REPO_ROOT / ".moto_update_notice.json"


def write_update_notice(result: UpdateCheckResult) -> None:
    """Persist a short update-notice payload so the backend can serve it as an in-app banner."""
    if not result.update_available or result.remote_manifest is None:
        cleanup_path(UPDATE_NOTICE_PATH)
        return

    payload = {
        "update_available": True,
        "installed_version": result.local_manifest.version,
        "installed_commit": result.local_manifest.short_commit,
        "available_version": result.remote_manifest.version,
        "available_commit": result.remote_manifest.short_commit,
        "install_layout": result.install_state.label,
        "can_auto_apply": result.can_apply_update,
        "message": build_warning_message(result) if not result.can_apply_update else build_update_prompt(result),
    }
    _write_json(UPDATE_NOTICE_PATH, payload)


def build_update_prompt(result: UpdateCheckResult) -> str:
    if result.remote_manifest is None:
        return "Update information is unavailable."

    return (
        "A newer MOTO build is available from GitHub main.\n\n"
        f"Installed: {result.local_manifest.version} ({result.local_manifest.short_commit})\n"
        f"Available: {result.remote_manifest.version} ({result.remote_manifest.short_commit})\n"
        f"Install layout: {result.install_state.label}\n\n"
        "If you continue, MOTO will preserve runtime data/log roots, instance-scoped local storage namespaces, and OS keyring namespaces."
    )


def build_warning_message(result: UpdateCheckResult) -> str:
    if result.remote_manifest is None:
        return result.error or "Update information is unavailable."

    details = []
    if result.warning:
        details.append(result.warning)
    if not result.install_state.can_auto_apply:
        details.append(result.install_state.reason)
    detail_text = "\n\n".join(details) if details else result.install_state.reason

    return (
        "A newer MOTO build is available from GitHub main, but this install cannot auto-apply it.\n\n"
        f"Installed: {result.local_manifest.version} ({result.local_manifest.short_commit})\n"
        f"Available: {result.remote_manifest.version} ({result.remote_manifest.short_commit})\n"
        f"Install layout: {result.install_state.label}\n\n"
        f"{detail_text}"
    )


def _relative_if_inside_repo(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    return str(relative).replace("\\", "/")


def _resolve_repo_relative_path(raw: str | None) -> Path | None:
    if not raw or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def collect_preserved_relatives(env: dict[str, str] | os._Environ[str], active_instances: list[dict] | None = None) -> set[str]:
    preserved = set(_DEFAULT_PRESERVED_ROOTS) | set(_DEFAULT_PRESERVED_FILES)

    active_instances = active_instances if active_instances is not None else cleanup_launcher_state()
    for instance in active_instances:
        if not isinstance(instance, dict):
            continue
        for key in ("data_root", "log_root"):
            value = str(instance.get(key, "")).strip()
            if not value:
                continue
            relative = _relative_if_inside_repo(Path(value))
            if relative:
                preserved.add(relative)

    for env_name in ("MOTO_DATA_ROOT", "MOTO_LOG_ROOT"):
        resolved = _resolve_repo_relative_path(env.get(env_name))
        if resolved:
            relative = _relative_if_inside_repo(resolved)
            if relative:
                preserved.add(relative)

    explicit_instance_id = _sanitize_instance_id(env.get("MOTO_INSTANCE_ID"))
    if explicit_instance_id:
        instance_root = REPO_ROOT / ".moto_instances" / explicit_instance_id
        relative = _relative_if_inside_repo(instance_root)
        if relative:
            preserved.add(relative)

    return {path.replace("\\", "/").strip("/") for path in preserved if path}


def _is_preserved(relative_path: str, preserved_relatives: set[str]) -> bool:
    normalized = relative_path.replace("\\", "/").strip("/")
    for preserved in preserved_relatives:
        if normalized == preserved or normalized.startswith(f"{preserved}/"):
            return True
    return False


def _ensure_backup_for_destination(destination: Path, backup_root: Path, relative_path: str, journal: _CopyJournal) -> None:
    backup_target = backup_root / relative_path
    if destination.is_dir():
        if relative_path in journal.overwritten_directories:
            return
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, backup_target, dirs_exist_ok=True)
        journal.overwritten_directories.append(relative_path)
        return

    if relative_path in journal.overwritten_files:
        return
    backup_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination, backup_target)
    journal.overwritten_files.append(relative_path)


def sync_snapshot_into_install(
    source_root: Path,
    destination_root: Path,
    preserved_relatives: set[str],
    backup_root: Path,
) -> _CopyJournal:
    journal = _CopyJournal()
    for source_path in source_root.rglob("*"):
        if source_path.is_dir():
            continue

        relative_path = str(source_path.relative_to(source_root)).replace("\\", "/")
        if _is_preserved(relative_path, preserved_relatives):
            continue

        destination_path = destination_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists():
            _ensure_backup_for_destination(destination_path, backup_root, relative_path, journal)
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
        else:
            journal.created_files.append(relative_path)

        shutil.copy2(source_path, destination_path)

    return journal


def restore_snapshot_from_backup(destination_root: Path, backup_root: Path, journal: _CopyJournal) -> None:
    for relative_path in reversed(journal.created_files):
        cleanup_path(destination_root / relative_path)
        _remove_empty_parents((destination_root / relative_path).parent)

    for relative_path in journal.overwritten_files:
        destination_path = destination_root / relative_path
        backup_path = backup_root / relative_path
        cleanup_path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, destination_path)

    for relative_path in journal.overwritten_directories:
        destination_path = destination_root / relative_path
        backup_path = backup_root / relative_path
        cleanup_path(destination_path)
        shutil.copytree(backup_path, destination_path, dirs_exist_ok=True)


def _remove_empty_parents(path: Path) -> None:
    current = path
    while current != REPO_ROOT:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _download_archive(manifest: BuildManifest, destination: Path) -> None:
    request = urllib.request.Request(
        archive_url_for_manifest(manifest),
        headers={"User-Agent": "MOTO-Build1-Updater"},
    )
    with urllib.request.urlopen(request, timeout=30) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)

    children = [child for child in destination.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination


def _resolve_entrypoint_path(raw: str | None) -> Path | None:
    if not raw or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def _build_relaunch_command(env: dict[str, str]) -> list[str]:
    entrypoint = _resolve_entrypoint_path(env.get(LAUNCHER_ENTRYPOINT_ENV))
    if entrypoint and entrypoint.exists():
        suffix = entrypoint.suffix.lower()
        if suffix == ".sh":
            return ["bash", str(entrypoint)]
        if suffix in {".bat", ".cmd"}:
            return ["cmd", "/c", str(entrypoint)]
        if suffix == ".ps1":
            return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(entrypoint)]
        return [str(entrypoint)]

    return [sys.executable, str(REPO_ROOT / "moto_launcher.py")]


def _relaunch_launcher(launcher_args: list[str], cleanup_paths: list[Path], env: dict[str, str]) -> None:
    command = _build_relaunch_command(env)
    command.extend(launcher_args)
    for cleanup_path in cleanup_paths:
        command.extend(["--moto-cleanup-update", str(cleanup_path)])
    subprocess.Popen(command, cwd=str(REPO_ROOT), env=env)


def apply_zip_update(
    *,
    remote_manifest: BuildManifest,
    launcher_args: list[str],
    env: dict[str, str],
) -> tuple[bool, str]:
    active_instances = cleanup_launcher_state()
    preserved_relatives = collect_preserved_relatives(env, active_instances)
    work_root = Path(tempfile.mkdtemp(prefix="moto-update-"))
    extract_root = work_root / "extract"
    archive_path = work_root / "update.zip"
    backup_root = Path(tempfile.mkdtemp(prefix="moto-update-backup-"))
    journal = _CopyJournal()

    try:
        _download_archive(remote_manifest, archive_path)
        extracted_source = _extract_archive(archive_path, extract_root)
        journal = sync_snapshot_into_install(extracted_source, REPO_ROOT, preserved_relatives, backup_root)
        _relaunch_launcher(launcher_args, [backup_root, work_root], env)
        return True, "Update applied successfully. Relaunching MOTO with the new build."
    except Exception as exc:
        restore_snapshot_from_backup(REPO_ROOT, backup_root, journal)
        cleanup_path(work_root)
        cleanup_path(backup_root)
        return False, f"Update failed and the previous install was restored: {exc}"


def _safe_fast_forward_possible() -> tuple[bool, str]:
    fetch_code, _, fetch_stderr = _git_output(["fetch", "origin", "main", "--quiet"])
    if fetch_code != 0:
        return False, fetch_stderr or "Failed to fetch origin/main."

    code, divergence_output, divergence_stderr = _git_output(["rev-list", "--left-right", "--count", "HEAD...origin/main"])
    if code != 0:
        return False, divergence_stderr or "Failed to compare HEAD against origin/main."

    try:
        ahead_str, behind_str = divergence_output.split()
        ahead = int(ahead_str)
        behind = int(behind_str)
    except (ValueError, TypeError):
        return False, "Failed to parse git divergence counts for origin/main."

    if ahead != 0:
        return False, "This checkout is ahead of origin/main and cannot be fast-forwarded automatically."
    if behind == 0:
        return False, "This checkout is already at the latest origin/main commit."
    return True, ""


def apply_git_update(
    *,
    launcher_args: list[str],
    env: dict[str, str],
) -> tuple[bool, str]:
    code, previous_head, stderr = _git_output(["rev-parse", "HEAD"])
    if code != 0 or not previous_head:
        return False, stderr or "Failed to determine the current git HEAD."

    safe_to_update, reason = _safe_fast_forward_possible()
    if not safe_to_update:
        return False, reason

    merge_code, _, merge_stderr = _git_output(["merge", "--ff-only", "origin/main"])
    if merge_code != 0:
        return False, merge_stderr or "Fast-forward merge from origin/main failed."

    try:
        _relaunch_launcher(launcher_args, [], env)
        return True, "Git checkout fast-forwarded successfully. Relaunching MOTO with the new build."
    except Exception as exc:
        _git_output(["reset", "--hard", previous_head])
        return False, f"Update was rolled back after relaunch failed: {exc}"


def apply_update(
    result: UpdateCheckResult,
    launcher_args: list[str],
    env: dict[str, str] | os._Environ[str],
) -> tuple[bool, str]:
    if result.remote_manifest is None:
        return False, result.error or "Update information is unavailable."
    if result.metadata_source != "manifest":
        return False, result.warning or "Automatic update-apply is disabled until the official main-branch manifest is published."

    env_copy = dict(env)
    if result.install_state.kind == "clean_git_clone":
        return apply_git_update(launcher_args=launcher_args, env=env_copy)
    if result.install_state.kind == "zip_install":
        return apply_zip_update(remote_manifest=result.remote_manifest, launcher_args=launcher_args, env=env_copy)
    return False, result.install_state.reason
