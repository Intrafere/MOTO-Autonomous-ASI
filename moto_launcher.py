"""
MOTO System Launcher (Python)
This is an internal script. Use "Click To Launch MOTO.bat" on Windows or "linux-ubuntu-launcher.sh" on Ubuntu 24.04.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import ntpath
import os
from pathlib import Path
import platform
from random import randint
import re
import secrets
import socket
import shlex
from shutil import copyfileobj, rmtree, which
import subprocess
import sys
import tarfile
import time
from typing import Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
import webbrowser
import zipfile

from moto_updater import (
    apply_update,
    build_update_prompt,
    check_for_updates,
    classify_install_state,
    cleanup_launcher_state,
    cleanup_path,
    consume_internal_launcher_args,
    load_last_instance_record,
    register_active_instance,
    save_last_instance_record,
    show_yes_no_dialog,
    write_update_notice,
)

SCRIPT_DIR = Path(__file__).resolve().parent

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
WHITE = "\033[97m"
RESET = "\033[0m"

MIN_PYTHON_VERSION = (3, 10)
MIN_NODE_VERSION = (20, 19, 0)
MIN_NODE_ALT_VERSION = (22, 12, 0)


@dataclass(frozen=True)
class InstanceRuntime:
    instance_id: str
    backend_host: str
    backend_port: int
    frontend_port: int
    data_root: str
    log_root: str
    secret_namespace: str | None
    storage_prefix: str | None
    is_default: bool
    # True when the caller used MOTO_INSTANCE_ID / MOTO_DATA_ROOT / etc. to
    # override the default runtime. Explicit launches are treated as one-off
    # overrides: we NEVER persist them to `.moto_last_instance.json`, so a
    # plain subsequent launch still points back at the user's stable default
    # / previously-recorded isolated instance.
    explicit_override: bool


@dataclass(frozen=True)
class LaunchedService:
    title: str
    pid: int
    mode: str
    log_path: str | None = None


def _enable_ansi_on_windows() -> None:
    if sys.platform != "win32":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except (AttributeError, OSError, ValueError):
        return


def cprint(message: str, colour: str = RESET) -> None:
    print(f"{colour}{message}{RESET}")


def exit_with_pause(code: int = 0) -> None:
    print()
    cprint("Press Enter to close...", YELLOW)
    with contextlib.suppress(EOFError):
        input()
    sys.exit(code)


def resolve_command(*names: str) -> str | None:
    for name in names:
        resolved = which(name)
        if resolved:
            return resolved
    return None


def resolve_existing_file(*paths: str | Path) -> str | None:
    for path in paths:
        try:
            candidate = Path(path).expanduser()
            if candidate.is_file():
                return str(candidate)
        except (OSError, RuntimeError):
            continue
    return None


def command_exists(name: str) -> bool:
    return resolve_command(name) is not None


def get_python_command() -> str:
    return sys.executable or resolve_command("python3", "python") or "python"


def parse_version_tuple(raw: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def format_version_tuple(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def node_version_is_supported(version: tuple[int, int, int]) -> bool:
    return (version[0] == 20 and version >= MIN_NODE_VERSION) or version >= MIN_NODE_ALT_VERSION


def _path_is_within(root: Path, candidate: str | Path) -> bool:
    try:
        Path(candidate).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _stored_keyring_namespace(record: dict | None) -> str | None:
    """Read current launcher state while accepting legacy records."""
    if not isinstance(record, dict):
        return None
    value = record.get("keyring_namespace")
    if value is None:
        value = record.get("secret_namespace")
    return value if isinstance(value, str) and value.strip() else None


def using_repo_local_venv() -> bool:
    return _path_is_within(SCRIPT_DIR / ".venv", get_python_command())


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def shell_join(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def get_standard_windows_node_file(filename: str) -> str | None:
    local_app_data = os.environ.get("LocalAppData")
    local_node_path = (
        Path(local_app_data) / "Programs" / "nodejs" / filename
        if local_app_data
        else Path.home() / "AppData" / "Local" / "Programs" / "nodejs" / filename
    )
    return resolve_existing_file(
        local_node_path,
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / filename,
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "nodejs" / filename,
    )


def refresh_windows_path_from_registry() -> None:
    """Refresh PATH after installers update Windows environment variables."""
    if sys.platform != "win32":
        return

    try:
        import winreg
    except ImportError:
        return

    registry_paths: list[str] = []
    keys = [
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ]
    for hive, subkey in keys:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if isinstance(value, str) and value.strip():
            registry_paths.extend(
                os.path.expandvars(part.strip())
                for part in value.split(os.pathsep)
                if part.strip()
            )

    for path_entry in reversed(registry_paths):
        prepend_process_path_entry(path_entry)


def prepend_process_path_entry(path_entry: str | Path | None) -> None:
    """Prepend a directory to this launcher's PATH if it is not already present."""
    if not path_entry:
        return

    try:
        normalized_entry = str(Path(path_entry).resolve())
    except OSError:
        normalized_entry = str(path_entry)

    current_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    normalized_parts = set()
    for part in current_parts:
        try:
            normalized_parts.add(str(Path(part).resolve()))
        except OSError:
            normalized_parts.add(part)

    if normalized_entry not in normalized_parts:
        os.environ["PATH"] = normalized_entry + os.pathsep + os.environ.get("PATH", "")


def ensure_windows_node_on_path(*commands: str | None) -> None:
    """Ensure npm child scripts can resolve plain `node` after a fresh install."""
    if sys.platform != "win32":
        return

    for filename in ("node.exe", "npm.cmd"):
        standard_path = get_standard_windows_node_file(filename)
        if standard_path:
            prepend_process_path_entry(Path(standard_path).parent)

    for command in commands:
        if not command:
            continue
        candidate = Path(command)
        if candidate.is_absolute() and candidate.parent.is_dir():
            prepend_process_path_entry(candidate.parent)


def get_node_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("node.exe", "node") or get_standard_windows_node_file("node.exe")
    return resolve_command("node")


def get_npm_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("npm.cmd", "npm.exe", "npm") or get_standard_windows_node_file("npm.cmd")
    return resolve_command("npm")


def get_lean_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("lean.exe", "lean")
    return resolve_command("lean")


def get_lake_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("lake.exe", "lake")
    return resolve_command("lake")


def get_z3_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("z3.exe", "z3")
    return resolve_command("z3")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        try:
            connection.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def find_free_port(start: int, exclude: set[int] | None = None) -> int:
    blocked_ports = exclude or set()
    for candidate in range(start, start + 2000):
        if candidate in blocked_ports:
            continue
        if not port_in_use(candidate):
            return candidate
    raise RuntimeError(f"Could not find a free port starting from {start}.")


def resolve_launcher_path(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    path = Path(raw)
    if path.is_absolute():
        return str(path.resolve())
    return str((SCRIPT_DIR / path).resolve())


def _runtime_lock_path(data_root: str) -> Path:
    return Path(data_root) / ".moto_runtime.lock"


def read_runtime_lock(data_root: str) -> dict[str, object]:
    try:
        payload = json.loads(_runtime_lock_path(data_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_runtime_lock_pid(data_root: str) -> int | None:
    payload = read_runtime_lock(data_root)
    if not payload:
        return None
    try:
        pid = int(payload.get("backend_pid"))
    except (AttributeError, TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def assert_runtime_lock_available(data_root: str) -> None:
    payload = read_runtime_lock(data_root)
    pid = read_runtime_lock_pid(data_root)
    if pid is not None and is_pid_running(pid):
        backend_port = int(payload.get("backend_port") or 0)
        if backend_port > 0 and not port_in_use(backend_port):
            try:
                _runtime_lock_path(data_root).unlink(missing_ok=True)
            except OSError:
                pass
            return
        raise RuntimeError(
            "The default MOTO data root is already in use by another backend process. "
            "Close the existing MOTO backend/frontend windows, then launch again."
        )


def write_runtime_lock(data_root: str, backend_pid: int, instance_id: str, backend_port: int | None = None) -> None:
    lock_path = _runtime_lock_path(data_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "backend_pid": backend_pid,
                "backend_port": backend_port,
                "instance_id": instance_id,
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def sanitize_instance_id(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("_")
    return normalized if normalized else None


def new_instance_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = randint(1000, 9999)
    return f"instance_{timestamp}_{suffix}"


def clear_console() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


def print_banner() -> None:
    cprint("================================================================", CYAN)
    cprint("  MOTO System Launcher", CYAN)
    cprint("================================================================", CYAN)
    print()


def resolve_instance_runtime() -> InstanceRuntime:
    explicit_id = sanitize_instance_id(os.environ.get("MOTO_INSTANCE_ID"))
    explicit_data = resolve_launcher_path(os.environ.get("MOTO_DATA_ROOT"))
    explicit_log = resolve_launcher_path(os.environ.get("MOTO_LOG_ROOT"))
    explicit_secret = sanitize_instance_id(os.environ.get("MOTO_SECRET_NAMESPACE"))
    explicit_storage = sanitize_instance_id(os.environ.get("MOTO_FRONTEND_STORAGE_PREFIX"))

    backend_host = os.environ.get("MOTO_BACKEND_HOST") or os.environ.get("HOST") or "127.0.0.1"

    explicit_backend_port = None
    for variable in ("MOTO_BACKEND_PORT", "PORT"):
        value = os.environ.get(variable)
        if value:
            explicit_backend_port = int(value)
            break

    requested_frontend_port = None
    for variable in ("MOTO_FRONTEND_PORT", "FRONTEND_PORT"):
        value = os.environ.get(variable)
        if value:
            requested_frontend_port = int(value)
            break

    default_data = str((SCRIPT_DIR / "backend" / "data").resolve())
    default_log = str((SCRIPT_DIR / "backend" / "logs").resolve())
    default_backend = 8000
    default_frontend = 5173

    has_explicit_identity = any(
        value is not None
        for value in (
            explicit_id,
            explicit_data,
            explicit_log,
            explicit_secret,
            explicit_storage,
        )
    )
    explicit_frontend_port = requested_frontend_port if has_explicit_identity else None

    # ------------------------------------------------------------------
    # CRITICAL: keyring/data/browser-storage stability across every plain
    # consumer relaunch.
    #
    # A normal launch must always point at the shared default runtime:
    # `backend/data`, `backend/logs`, no keyring namespace, and no frontend
    # storage prefix. Port availability is not user identity. If backend port
    # 8000 is busy, only the backend port may move. The default frontend origin
    # stays on 5173 so browser profiles/prompts remain visible.
    # Isolated `.moto_instances/*` roots are reserved for explicit identity /
    # storage overrides. Port-only overrides are not identity overrides.
    # ------------------------------------------------------------------
    reused_record: dict | None = None
    active_plain_instance_ids: set[str] = set()
    if not has_explicit_identity:
        active_plain_instance_ids = {
            str(active.get("instance_id") or "").strip()
            for active in cleanup_launcher_state()
            if isinstance(active, dict)
        }
        if "default" in active_plain_instance_ids:
            raise RuntimeError(
                "The default MOTO instance already appears to be running. Close the existing "
                "MOTO backend/frontend windows, then launch again. A plain launch will not "
                "create a separate empty data/keyring namespace."
            )

        last_record = load_last_instance_record()
        if last_record is not None:
            candidate_id = sanitize_instance_id(last_record.get("instance_id")) or "default"
            if candidate_id == "default":
                reused_record = {
                    "instance_id": candidate_id,
                    "data_root": None,
                    "log_root": None,
                    "keyring_namespace": None,
                    "storage_prefix": None,
                }

    # Decide the instance identity.
    if has_explicit_identity:
        instance_id = explicit_id or new_instance_id()
    elif reused_record is not None:
        instance_id = reused_record["instance_id"]
    else:
        # No stored runtime yet. Adopt the shared default identity; later
        # backend port selection may choose a free port. Do not make a plain
        # relaunch look like a brand-new user.
        instance_id = "default"

    is_default_instance = instance_id == "default"

    # Resolve data / log roots.
    if is_default_instance:
        data_root = explicit_data or (reused_record or {}).get("data_root") or default_data
        log_root = explicit_log or (reused_record or {}).get("log_root") or default_log
    else:
        instance_root = (SCRIPT_DIR / ".moto_instances" / instance_id).resolve()
        data_root = (
            explicit_data
            or (reused_record or {}).get("data_root")
            or str(instance_root / "data")
        )
        log_root = (
            explicit_log
            or (reused_record or {}).get("log_root")
            or str(instance_root / "logs")
        )

    # Resolve ports. We always pick a free port; ports are not part of the
    # keyring namespace, so changing them between launches is safe.
    if explicit_backend_port is not None:
        if port_in_use(explicit_backend_port):
            raise RuntimeError(f"Requested backend port {explicit_backend_port} is already in use.")
        backend_port = explicit_backend_port
    else:
        backend_port = default_backend if not port_in_use(default_backend) else find_free_port(default_backend)

    if explicit_frontend_port is not None:
        if explicit_frontend_port == backend_port:
            raise RuntimeError(f"Frontend port cannot match backend port ({backend_port}).")
        if port_in_use(explicit_frontend_port):
            raise RuntimeError(f"Requested frontend port {explicit_frontend_port} is already in use.")
        frontend_port = explicit_frontend_port
    elif requested_frontend_port is not None and is_default_instance and not has_explicit_identity:
        raise RuntimeError(
            "Frontend port overrides are disabled for normal default launches because changing "
            "the localhost port hides browser-saved profiles/prompts. Close the process using "
            f"port {default_frontend}, or use explicit instance/data-root overrides for an isolated run."
        )
    elif is_default_instance and not has_explicit_identity and port_in_use(default_frontend):
        raise RuntimeError(
            f"Frontend port {default_frontend} is already in use. Close the existing MOTO "
            "frontend/browser launch and start again so saved browser profiles/prompts stay "
            "on the normal localhost:5173 origin."
        )
    else:
        if not port_in_use(default_frontend) and default_frontend != backend_port:
            frontend_port = default_frontend
        else:
            frontend_port = find_free_port(default_frontend, exclude={backend_port})

    # Resolve secret namespace / storage prefix.
    # Default instance: explicit shared "no namespace" → keyring service
    # name has no suffix (legacy). Non-default instance: namespace is the
    # instance_id unless explicitly overridden or reused from a record that
    # stored an explicit override.
    if is_default_instance:
        secret_namespace = explicit_secret or _stored_keyring_namespace(reused_record)
        storage_prefix = explicit_storage or (reused_record or {}).get("storage_prefix")
    else:
        recorded_secret = _stored_keyring_namespace(reused_record)
        recorded_storage = (reused_record or {}).get("storage_prefix")
        secret_namespace = (
            explicit_secret
            or (sanitize_instance_id(recorded_secret) if recorded_secret else None)
            or instance_id
        )
        storage_prefix = (
            explicit_storage
            or (sanitize_instance_id(recorded_storage) if recorded_storage else None)
            or instance_id
        )

    return InstanceRuntime(
        instance_id=instance_id,
        backend_host=backend_host,
        backend_port=backend_port,
        frontend_port=frontend_port,
        data_root=str(Path(data_root).resolve()),
        log_root=str(Path(log_root).resolve()),
        secret_namespace=secret_namespace,
        storage_prefix=storage_prefix,
        is_default=is_default_instance,
        explicit_override=has_explicit_identity,
    )


def run_visible(args: list[str], cwd: str | None = None, check: bool = True) -> int:
    result = subprocess.run(args, cwd=cwd, check=False)
    if check and result.returncode != 0:
        return result.returncode
    return result.returncode


def run_silent(args: list[str], cwd: str | None = None) -> int:
    return subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode


def has_desktop_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_linux_terminal() -> tuple[str, str] | None:
    if not is_linux() or not has_desktop_session():
        return None
    for candidate in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
        resolved = resolve_command(candidate)
        if resolved:
            return candidate, resolved
    return None


def build_linux_terminal_command(
    terminal_name: str,
    terminal_path: str,
    title: str,
    args: Sequence[str],
    cwd: str,
) -> list[str]:
    command_text = f"cd {shlex.quote(cwd)} && exec {shell_join(args)}"
    if terminal_name == "gnome-terminal":
        return [terminal_path, f"--title={title}", "--", "bash", "-lc", command_text]
    if terminal_name == "konsole":
        return [terminal_path, "-p", f"tabtitle={title}", "-e", "bash", "-lc", command_text]
    return [terminal_path, "-T", title, "-e", "bash", "-lc", command_text]


def resolve_windows_console_executable(executable: str) -> str:
    """Prefer a PATH-safe executable name when building cmd.exe command text."""
    candidate = str(executable or "").strip()
    if sys.platform != "win32" or not candidate or not ntpath.isabs(candidate):
        return candidate

    command_name = ntpath.basename(candidate)
    resolved = resolve_command(command_name)
    if not resolved:
        return candidate

    if ntpath.normcase(ntpath.normpath(resolved)) == ntpath.normcase(ntpath.normpath(candidate)):
        return command_name

    return candidate


def windows_service_requires_direct_launch(args: Sequence[str]) -> bool:
    """Return True when cmd.exe quoting would be unsafe for this command."""
    if sys.platform != "win32" or not args:
        return False

    executable = str(args[0] or "").strip()
    if not executable or not ntpath.isabs(executable):
        return False

    normalized = resolve_windows_console_executable(executable)
    return normalized == executable and any(character.isspace() for character in executable)


def build_windows_service_command(title: str, args: Sequence[str]) -> str:
    """Build a cmd.exe-safe command string for a titled service window."""
    shell_args = list(args)
    if shell_args:
        shell_args[0] = resolve_windows_console_executable(str(shell_args[0]))
    return f"title {title} && {subprocess.list2cmdline(shell_args)}"


def launch_windows_service(title: str, args: Sequence[str], cwd: str, env: dict[str, str]) -> LaunchedService:
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    # Some Windows tools (notably npm.cmd under "Program Files") break when a
    # quoted absolute path is embedded inside a `cmd /k` string. Prefer the
    # PATH-safe executable name when possible; otherwise launch directly.
    if windows_service_requires_direct_launch(args):
        process = subprocess.Popen(
            list(args),
            cwd=cwd,
            env=env,
            creationflags=creationflags,
        )
        return LaunchedService(title=title, pid=process.pid, mode="window")

    process = subprocess.Popen(
        ["cmd", "/k", build_windows_service_command(title, args)],
        cwd=cwd,
        env=env,
        creationflags=creationflags,
    )
    return LaunchedService(title=title, pid=process.pid, mode="window")


def launch_linux_terminal_service(
    title: str,
    args: Sequence[str],
    cwd: str,
    env: dict[str, str],
) -> LaunchedService | None:
    terminal = resolve_linux_terminal()
    if terminal is None:
        return None
    terminal_name, terminal_path = terminal
    try:
        process = subprocess.Popen(
            build_linux_terminal_command(terminal_name, terminal_path, title, args, cwd),
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except OSError:
        return None
    return LaunchedService(title=title, pid=process.pid, mode="terminal")


def launch_background_service(
    title: str,
    service_slug: str,
    args: Sequence[str],
    cwd: str,
    env: dict[str, str],
    log_root: str,
) -> LaunchedService:
    log_path = Path(log_root) / f"launcher_{service_slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n=== {title} ({datetime.now().isoformat(timespec='seconds')}) ===\n")
    stream.write(f"Command: {shell_join(args)}\n\n")
    stream.flush()
    process = subprocess.Popen(
        list(args),
        cwd=cwd,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return LaunchedService(title=title, pid=process.pid, mode="background", log_path=str(log_path))


def launch_service(
    title: str,
    service_slug: str,
    args: Sequence[str],
    cwd: str,
    env: dict[str, str],
    log_root: str,
) -> LaunchedService:
    if sys.platform == "win32":
        return launch_windows_service(title, args, cwd, env)
    if is_linux():
        terminal_service = launch_linux_terminal_service(title, args, cwd, env)
        if terminal_service is not None:
            return terminal_service
    return launch_background_service(title, service_slug, args, cwd, env, log_root)


def cleanup_relaunch_artifacts(cleanup_paths: list[Path]) -> None:
    if not cleanup_paths:
        return
    for cleanup_target in cleanup_paths:
        cleanup_path(cleanup_target)


def handle_available_updates(launcher_args: list[str]) -> bool:
    cprint("[Update] Checking for main-branch updates...", YELLOW)
    result = check_for_updates()

    if result.install_state.active_instance_count:
        count = result.install_state.active_instance_count
        suffix = "s" if count != 1 else ""
        cprint(f"[Update] Detected {count} launcher-managed instance{suffix} from this install.", YELLOW)

    if result.error:
        cprint(f"[Update] Skipping update check: {result.error}", YELLOW)
        write_update_notice(result)
        print()
        return True

    if result.warning:
        cprint(f"[Update] {result.warning}", YELLOW)

    if not result.update_available:
        write_update_notice(result)
        cprint(
            f"[Update] Launcher is already on {result.local_manifest.version} ({result.local_manifest.short_commit}).",
            GREEN,
        )
        print()
        return True

    remote_manifest = result.remote_manifest
    assert remote_manifest is not None
    cprint(
        f"[Update] Found {remote_manifest.version} ({remote_manifest.short_commit}) on GitHub main.",
        YELLOW,
    )

    if not result.can_apply_update:
        if not result.warning:
            cprint(f"[Update] {result.install_state.reason}", YELLOW)
        write_update_notice(result)
        cprint("[Update] Update notice saved — the app will show an in-app banner.", YELLOW)
        print()
        return True

    if not show_yes_no_dialog("MOTO Update Available", build_update_prompt(result)):
        cprint("[Update] Continuing without applying the update.", YELLOW)
        write_update_notice(result)
        print()
        return True

    applied, message = apply_update(result, launcher_args, os.environ)
    cprint(f"[Update] {message}", GREEN if applied else YELLOW)
    print()
    return not applied


def check_python_installation() -> None:
    cprint("[1/8] Checking Python installation...", YELLOW)
    python_cmd = get_python_command()
    if not python_cmd:
        print()
        cprint("============================================================", RED)
        cprint(f"ERROR: Python {format_version_tuple(MIN_PYTHON_VERSION)}+ is required to run the launcher", RED)
        cprint("============================================================", RED)
        print()
        if is_linux():
            cprint("Install Python 3 and python3-venv, then launch via `bash linux-ubuntu-launcher.sh`.", YELLOW)
            cprint("Example: sudo apt install python3 python3-venv", YELLOW)
        else:
            cprint(f"Please install Python {format_version_tuple(MIN_PYTHON_VERSION)}+ from:", YELLOW)
            cprint("https://www.python.org/downloads/", YELLOW)
            print()
            cprint("IMPORTANT: Check 'Add Python to PATH' during installation", YELLOW)
        exit_with_pause(1)

    version = subprocess.check_output([python_cmd, "--version"], text=True).strip()
    if sys.version_info < MIN_PYTHON_VERSION:
        print()
        cprint("============================================================", RED)
        cprint(f"ERROR: Python {format_version_tuple(MIN_PYTHON_VERSION)}+ is required", RED)
        cprint("============================================================", RED)
        print()
        cprint(f"Current interpreter: {version} ({python_cmd})", YELLOW)
        if is_linux():
            cprint("Install a newer Python and relaunch via `bash linux-ubuntu-launcher.sh`.", YELLOW)
        else:
            cprint("The Windows one-click launcher can install Python 3.12 automatically when Python is missing.", YELLOW)
            cprint("If you launched this script directly, install Python 3.12 or double-click `Click To Launch MOTO.bat`.", YELLOW)
        exit_with_pause(1)
    cprint(version, GREEN)
    cprint(f"Interpreter: {python_cmd}", WHITE)
    if is_linux():
        if using_repo_local_venv():
            cprint("Using repo-local .venv for Ubuntu-safe package installs.", GREEN)
        else:
            cprint("Tip: `linux-ubuntu-launcher.sh` is the recommended Ubuntu 24.04 entrypoint because it keeps Python packages inside the repo-local .venv.", YELLOW)
    print()


def install_windows_nodejs() -> bool:
    if sys.platform != "win32":
        return False

    winget_cmd = resolve_command("winget.exe", "winget")
    if not winget_cmd:
        return False

    cprint("Attempting to install Node.js LTS with winget...", YELLOW)
    run_visible(
        [winget_cmd, "source", "update", "--name", "winget"],
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    base_command = [
        winget_cmd,
        "install",
        "--id",
        "",
        "-e",
        "--source",
        "winget",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    package_ids = ("OpenJS.NodeJS.LTS", "OpenJS.NodeJS")
    scope_args = (["--scope", "user"], [])

    for package_id in package_ids:
        for scope_arg in scope_args:
            command = list(base_command)
            command[3] = package_id
            command.extend(scope_arg)
            scope_label = "user scope" if scope_arg else "default scope"
            cprint(f"Trying winget package {package_id} ({scope_label})...", YELLOW)
            if run_visible(command, cwd=str(SCRIPT_DIR), check=False) == 0:
                return True

    return False


def check_node_installation() -> None:
    cprint("[2/8] Checking Node.js installation...", YELLOW)
    node_cmd = get_node_command()
    if not node_cmd:
        if sys.platform == "win32" and install_windows_nodejs():
            refresh_windows_path_from_registry()
            node_cmd = get_node_command()
        if not node_cmd:
            print()
            cprint("============================================================", RED)
            cprint("ERROR: Node.js is not installed or not in PATH", RED)
            cprint("============================================================", RED)
            print()
            if is_linux():
                cprint("Install Node.js 20.19+ or 22.12+ from nodejs.org or your Ubuntu package source, then retry.", YELLOW)
            else:
                cprint("Please install Node.js 20.19+ or 22.12+ from:", YELLOW)
                cprint("https://nodejs.org/", YELLOW)
                cprint("The Windows launcher tried winget packages `OpenJS.NodeJS.LTS` and `OpenJS.NodeJS`, but they were unavailable or failed.", YELLOW)
            exit_with_pause(1)

    npm_cmd = get_npm_command()
    if not npm_cmd:
        if sys.platform == "win32":
            refresh_windows_path_from_registry()
            npm_cmd = get_npm_command()
    if not npm_cmd:
        print()
        cprint("============================================================", RED)
        cprint("ERROR: npm is not available in PATH", RED)
        cprint("============================================================", RED)
        print()
        cprint("Node.js appears to be installed, but npm could not be found.", YELLOW)
        if is_linux():
            cprint("Reinstall Node.js and ensure both `node` and `npm` are available in PATH.", YELLOW)
        else:
            cprint("Reinstall Node.js from https://nodejs.org/ and ensure npm is included in PATH.", YELLOW)
        exit_with_pause(1)

    node_version = subprocess.check_output([node_cmd, "--version"], text=True).strip()
    parsed_node_version = parse_version_tuple(node_version)
    if not parsed_node_version or not node_version_is_supported(parsed_node_version):
        if sys.platform == "win32" and install_windows_nodejs():
            refresh_windows_path_from_registry()
            node_cmd = get_standard_windows_node_file("node.exe") or get_node_command() or node_cmd
            npm_cmd = get_standard_windows_node_file("npm.cmd") or get_npm_command() or npm_cmd
            node_version = subprocess.check_output([node_cmd, "--version"], text=True).strip()
            parsed_node_version = parse_version_tuple(node_version)

    if not parsed_node_version or not node_version_is_supported(parsed_node_version):
        print()
        cprint("============================================================", RED)
        cprint("ERROR: Node.js 20.19+ or 22.12+ is required", RED)
        cprint("============================================================", RED)
        print()
        cprint(f"Current Node.js version: {node_version}", YELLOW)
        if is_linux():
            cprint("Install Node.js 20.19+ or 22.12+ from nodejs.org or your Ubuntu package source, then retry.", YELLOW)
        else:
            cprint("Install Node.js LTS from https://nodejs.org/ or rerun the launcher after winget is available.", YELLOW)
        exit_with_pause(1)

    npm_version = subprocess.check_output([npm_cmd, "--version"], text=True).strip()
    ensure_windows_node_on_path(node_cmd, npm_cmd)
    cprint(f"Node: {node_version}", GREEN)
    cprint(f"npm: {npm_version}", GREEN)
    print()


def prepare_runtime_and_environment() -> tuple[InstanceRuntime, str, str, dict[str, str]]:
    cprint("[3/8] Resolving instance runtime...", YELLOW)
    runtime = resolve_instance_runtime()
    frontend_url = f"http://localhost:{runtime.frontend_port}"
    backend_url = f"http://localhost:{runtime.backend_port}"
    user_uploads = os.path.join(runtime.data_root, "user_uploads")

    for directory in (runtime.data_root, runtime.log_root, user_uploads):
        os.makedirs(directory, exist_ok=True)

    last_record = load_last_instance_record()
    reused_from_record = (
        last_record is not None
        and sanitize_instance_id(last_record.get("instance_id")) == runtime.instance_id
        and not runtime.explicit_override
    )

    if runtime.explicit_override:
        cprint(
            "Explicit instance overrides detected (MOTO_INSTANCE_ID / MOTO_DATA_ROOT / "
            "MOTO_SECRET_NAMESPACE). This launch will NOT update the persisted last-instance "
            "record so a plain relaunch still points back at your default setup.",
            YELLOW,
        )
    elif runtime.is_default:
        if reused_from_record:
            cprint(
                "Reusing the default instance runtime (shared keyring namespace, stable data root).",
                GREEN,
            )
        else:
            cprint("Using default instance storage (shared keyring namespace).", GREEN)
    else:
        if reused_from_record:
            cprint(
                "Reusing previously launched instance runtime (same keyring namespace, same data root).",
                GREEN,
            )
        else:
            cprint("Launching an isolated instance with its own data root and keyring namespace.", GREEN)

    cprint(f"Instance ID: {runtime.instance_id}", CYAN)
    cprint(f"Backend URL: {backend_url}", GREEN)
    cprint(f"Frontend URL: {frontend_url}", GREEN)
    cprint(f"Data root: {runtime.data_root}", WHITE)
    cprint(f"Log root: {runtime.log_root}", WHITE)
    if runtime.secret_namespace:
        cprint("Keyring namespace: configured for this instance", WHITE)
    else:
        cprint("Keyring namespace: shared default store", WHITE)
    print()

    env = os.environ.copy()
    env["MOTO_INSTANCE_ID"] = runtime.instance_id
    env["MOTO_DATA_ROOT"] = runtime.data_root
    env["MOTO_LOG_ROOT"] = runtime.log_root
    env["MOTO_BACKEND_HOST"] = runtime.backend_host
    env["HOST"] = runtime.backend_host
    env["MOTO_BACKEND_PORT"] = str(runtime.backend_port)
    env["PORT"] = str(runtime.backend_port)
    env["MOTO_FRONTEND_PORT"] = str(runtime.frontend_port)
    env["FRONTEND_PORT"] = str(runtime.frontend_port)
    env["VITE_MOTO_FRONTEND_PORT"] = str(runtime.frontend_port)
    env["VITE_MOTO_BACKEND_URL"] = backend_url
    env["VITE_MOTO_INSTANCE_ID"] = runtime.instance_id
    env["VITE_MOTO_DATA_ROOT_DISPLAY"] = runtime.data_root
    desktop_api_token = os.environ.get("MOTO_DESKTOP_API_TOKEN") or secrets.token_urlsafe(32)
    env["MOTO_DESKTOP_API_TOKEN"] = desktop_api_token
    env["VITE_MOTO_DESKTOP_API_TOKEN"] = desktop_api_token

    if runtime.storage_prefix:
        env["MOTO_FRONTEND_STORAGE_PREFIX"] = runtime.storage_prefix
        env["VITE_MOTO_STORAGE_PREFIX"] = runtime.storage_prefix
    else:
        env.pop("MOTO_FRONTEND_STORAGE_PREFIX", None)
        env.pop("VITE_MOTO_STORAGE_PREFIX", None)

    if runtime.secret_namespace:
        env["MOTO_SECRET_NAMESPACE"] = runtime.secret_namespace
    else:
        env.pop("MOTO_SECRET_NAMESPACE", None)

    return runtime, frontend_url, backend_url, env


def install_python_dependencies() -> None:
    cprint("[4/8] Installing Python dependencies...", YELLOW)
    cprint("Upgrading pip and checking packages...", YELLOW)
    print()
    python_cmd = get_python_command()
    run_silent([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], cwd=str(SCRIPT_DIR))
    result = run_visible(
        [python_cmd, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"],
        cwd=str(SCRIPT_DIR),
        check=False,
    )
    if result != 0:
        print()
        cprint("============================================================", RED)
        cprint("ERROR: Failed to install Python dependencies", RED)
        cprint("============================================================", RED)
        print()
        cprint("Please check:", YELLOW)
        cprint("- Internet connection is working", YELLOW)
        cprint("- You have permission to install packages", YELLOW)
        if is_linux():
            cprint("- On Ubuntu 24.04, prefer launching via `bash linux-ubuntu-launcher.sh` so installs stay inside the repo-local .venv", YELLOW)
            cprint("- If venv creation fails, install `python3-venv` first", YELLOW)
        exit_with_pause(1)
    cprint("Python dependencies up to date", GREEN)
    print()


def install_playwright_browser() -> None:
    cprint("[4b/8] Installing Playwright Chromium browser for PDF generation...", YELLOW)
    cprint("This is a one-time download (~150MB) and may take a few minutes...", YELLOW)
    print()
    python_cmd = get_python_command()
    result = run_visible([python_cmd, "-m", "playwright", "install", "chromium"], cwd=str(SCRIPT_DIR), check=False)
    if result != 0:
        print()
        cprint("WARNING: Playwright Chromium install failed.", YELLOW)
        cprint("PDF generation will not be available until resolved.", YELLOW)
        cprint(f"Retry manually: {python_cmd} -m playwright install chromium", YELLOW)
        if is_linux():
            cprint("Ubuntu 24.04 may also require desktop/browser system libraries before Playwright can launch Chromium successfully.", YELLOW)
        cprint("Continuing startup anyway...", YELLOW)
    else:
        cprint("Playwright Chromium ready!", GREEN)
    print()


def _prepend_path_entry(path_entry: str, env: dict[str, str]) -> None:
    """Prepend a directory to PATH for the current process and child services."""
    if not path_entry:
        return
    prepend_process_path_entry(path_entry)
    env["PATH"] = os.environ.get("PATH", "")


def _write_lean_workspace_files(workspace_dir: Path) -> None:
    """Create the reusable Lean 4 Mathlib workspace files."""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    lean_toolchain_path = workspace_dir / "lean-toolchain"
    if not lean_toolchain_path.exists():
        lean_toolchain_path.write_text("leanprover/lean4:stable\n", encoding="utf-8")

    lakefile_path = workspace_dir / "lakefile.lean"
    if not lakefile_path.exists():
        lakefile_path.write_text(
            "\n".join(
                [
                    "import Lake",
                    "open Lake DSL",
                    "",
                    "package «moto_proof_workspace» where",
                    "",
                    "require mathlib from git",
                    '  "https://github.com/leanprover-community/mathlib4.git"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    root_file_path = workspace_dir / "MOTOProofWorkspace.lean"
    if not root_file_path.exists():
        root_file_path.write_text("import Mathlib\n", encoding="utf-8")


def _download_file(url: str, destination: Path) -> None:
    """Download a remote file to disk using the standard library.

    Writes to a sibling temp file first, then atomically renames on success so
    a partial download caused by a timeout or interruption never leaves a
    corrupt file at the destination path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        request = Request(url, headers={"User-Agent": "MOTO Launcher"})
        with urlopen(request) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(destination)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract a zip or tarball into the destination directory."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    def ensure_member_target(member_name: str) -> None:
        target = (destination_root / member_name).resolve()
        try:
            target.relative_to(destination_root)
        except ValueError as exc:
            raise RuntimeError(f"Archive member escapes destination: {member_name}") from exc

    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                ensure_member_target(member.filename)
            archive.extractall(destination)
        return
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                ensure_member_target(member.name)
                target = (destination_root / member.name).resolve()
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(f"Unsupported archive member type: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member: {member.name}")
                with source, target.open("wb") as output:
                    copyfileobj(source, output)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _is_valid_archive(archive_path: Path) -> bool:
    """Return True when the file at archive_path is a readable zip or tarball."""
    try:
        name = archive_path.name.lower()
        if name.endswith(".zip"):
            return zipfile.is_zipfile(archive_path)
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            return tarfile.is_tarfile(archive_path)
        return False
    except OSError:
        return False


def _detect_z3_asset_name() -> tuple[str, tuple[str, ...]]:
    """Return the preferred platform marker and fallback markers for Z3 assets."""
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch_markers = ("x64",)
    elif machine in {"aarch64", "arm64"}:
        arch_markers = ("arm64", "aarch64")
    else:
        raise RuntimeError(f"Unsupported architecture for automatic Z3 install: {machine}")

    if sys.platform == "win32":
        return "win", arch_markers
    if is_linux():
        return "glibc", arch_markers
    raise RuntimeError(f"Automatic Z3 install is unsupported on platform: {sys.platform}")


def _select_z3_asset(assets: list[dict[str, str]]) -> dict[str, str]:
    """Pick the best-matching release asset for the current platform."""
    platform_marker, arch_markers = _detect_z3_asset_name()
    candidates = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("name")
        and asset.get("browser_download_url")
        and ".sig" not in asset["name"].lower()
        and any(marker in asset["name"].lower() for marker in arch_markers)
    ]

    for asset in candidates:
        name = asset["name"].lower()
        if platform_marker in name and (name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz")):
            return asset

    readable_assets = ", ".join(asset.get("name", "<unknown>") for asset in assets if isinstance(asset, dict))
    raise RuntimeError(f"Could not find a matching Z3 release asset. Available assets: {readable_assets}")


def _find_z3_binary(search_root: Path) -> Path | None:
    """Locate the extracted Z3 executable."""
    candidate_names = ("z3.exe", "z3") if sys.platform == "win32" else ("z3",)
    for candidate_name in candidate_names:
        for candidate in search_root.rglob(candidate_name):
            if candidate.is_file():
                return candidate
    return None


def _select_elan_windows_asset(assets: list[dict[str, str]]) -> dict[str, str]:
    """Pick the best-matching Windows elan release asset for the current architecture.

    Upstream elan no longer publishes a bare ``elan-init.exe``; instead it ships
    per-platform zip archives (for example ``elan-x86_64-pc-windows-msvc.zip``)
    that contain ``elan-init.exe`` inside. We prefer the native architecture and
    fall back to x86_64 so ARM64 Windows hosts can still bootstrap via emulation.
    """
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        preferred_markers = ("aarch64-pc-windows", "arm64-pc-windows")
        fallback_markers = ("x86_64-pc-windows", "amd64-pc-windows")
    else:
        preferred_markers = ("x86_64-pc-windows", "amd64-pc-windows")
        fallback_markers: tuple[str, ...] = ()

    candidates = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("name")
        and asset.get("browser_download_url")
        and asset["name"].lower().endswith(".zip")
        and ".sig" not in asset["name"].lower()
    ]

    for markers in (preferred_markers, fallback_markers):
        for asset in candidates:
            name = asset["name"].lower()
            if any(marker in name for marker in markers):
                return asset

    readable_assets = ", ".join(asset.get("name", "<unknown>") for asset in assets if isinstance(asset, dict))
    raise RuntimeError(
        f"Could not find a matching elan release asset for Windows. Available assets: {readable_assets}"
    )


def _find_elan_installer(search_root: Path) -> Path | None:
    """Locate the extracted elan-init installer executable."""
    if not search_root.exists():
        return None
    candidate_names = ("elan-init.exe",) if sys.platform == "win32" else ("elan-init",)
    for candidate_name in candidate_names:
        for candidate in search_root.rglob(candidate_name):
            if candidate.is_file():
                return candidate
    return None


def _set_lean_env_flags(
    env: dict[str, str],
    *,
    enabled: bool,
    lean_path: str = "",
    workspace_dir: str = "",
) -> None:
    env["MOTO_LEAN4_ENABLED"] = "1" if enabled else "0"
    env["MOTO_LEAN4_PATH"] = lean_path
    env["MOTO_LEAN4_WORKSPACE_DIR"] = workspace_dir
    env["MOTO_LEAN4_PROOF_TIMEOUT"] = env.get("MOTO_LEAN4_PROOF_TIMEOUT", "").strip() or "900"
    env["MOTO_LEAN4_LSP_ENABLED"] = (
        env.get("MOTO_LEAN4_LSP_ENABLED", "").strip()
        if enabled and env.get("MOTO_LEAN4_LSP_ENABLED", "").strip()
        else ("1" if enabled else "0")
    )
    env["MOTO_LEAN4_LSP_IDLE_TIMEOUT"] = env.get("MOTO_LEAN4_LSP_IDLE_TIMEOUT", "").strip() or "600"


def _set_smt_env_flags(
    env: dict[str, str],
    *,
    enabled: bool,
    z3_path: str = "",
) -> None:
    env["MOTO_SMT_ENABLED"] = "1" if enabled else "0"
    env["MOTO_Z3_PATH"] = z3_path
    env["MOTO_SMT_TIMEOUT"] = env.get("MOTO_SMT_TIMEOUT", "").strip() or "30"


def install_lean4(
    runtime: InstanceRuntime,
    env: dict[str, str],
    *,
    _is_repair: bool = False,
) -> None:
    """
    Ensure Lean 4 / elan is available for proof verification.

    This step is intentionally non-fatal: if installation fails, MOTO still
    launches and simply skips automated proof verification.
    """
    if not _is_repair:
        cprint("[4c/8] Checking Lean 4 / elan for proof verification...", YELLOW)
        print()

    elan_bin_dir = Path.home() / ".elan" / "bin"
    lean_cmd = get_lean_command()

    if lean_cmd and elan_bin_dir.exists():
        _prepend_path_entry(str(elan_bin_dir), env)

    if not lean_cmd:
        cprint("Lean 4 not detected. Attempting one-time elan installation...", YELLOW)
        try:
            if sys.platform == "win32":
                managed_root = Path(runtime.data_root) / "elan"
                release_request = Request(
                    "https://api.github.com/repos/leanprover/elan/releases/latest",
                    headers={
                        "User-Agent": "MOTO Launcher",
                        "Accept": "application/vnd.github+json",
                    },
                )
                with urlopen(release_request, timeout=60) as response:
                    release_payload = json.loads(response.read().decode("utf-8"))

                asset = _select_elan_windows_asset(list(release_payload.get("assets") or []))
                archive_path = managed_root / "downloads" / asset["name"]
                install_root = managed_root / "current"

                if archive_path.exists() and not _is_valid_archive(archive_path):
                    cprint("Cached elan archive appears corrupt — re-downloading...", YELLOW)
                    archive_path.unlink(missing_ok=True)

                if not archive_path.exists():
                    _download_file(asset["browser_download_url"], archive_path)

                installer_path = _find_elan_installer(install_root)
                if installer_path is None:
                    _extract_archive(archive_path, install_root)
                    installer_path = _find_elan_installer(install_root)

                if installer_path is None:
                    raise RuntimeError(
                        "elan archive extracted successfully, but elan-init.exe could not be located."
                    )

                install_result = run_visible(
                    [
                        str(installer_path),
                        "-y",
                        "--default-toolchain",
                        "leanprover/lean4:stable",
                    ],
                    cwd=str(SCRIPT_DIR),
                    check=False,
                )
                if install_result != 0:
                    raise RuntimeError("elan installer exited with a non-zero status.")
            else:
                install_result = subprocess.run(
                    [
                        "sh",
                        "-c",
                        "curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:stable",
                    ],
                    cwd=str(SCRIPT_DIR),
                    check=False,
                ).returncode
                if install_result != 0:
                    raise RuntimeError("elan install script exited with a non-zero status.")
        except Exception as exc:
            print()
            cprint("WARNING: Lean 4 not available -- proof verification will be skipped.", YELLOW)
            cprint(str(exc), YELLOW)
            _set_lean_env_flags(env, enabled=False)
            print()
            return

    if elan_bin_dir.exists():
        _prepend_path_entry(str(elan_bin_dir), env)

    lean_cmd = get_lean_command()
    lake_cmd = get_lake_command()
    if not lean_cmd or not lake_cmd:
        # Tooling is incomplete (e.g. lake missing after a partial elan update).
        # Attempt the same wipe-and-retry repair before giving up.
        cprint("Lean 4 tooling is incomplete (lean or lake not found after install).", YELLOW)
        if env.get("_MOTO_LEAN4_REPAIR_ATTEMPTED") != "1":
            cprint("Attempting to repair by wiping elan and reinstalling...", YELLOW)
            print()
            try:
                if elan_bin_dir.parent.exists():
                    rmtree(str(elan_bin_dir.parent), ignore_errors=True)
                managed_elan_root = Path(runtime.data_root) / "elan"
                if managed_elan_root.exists():
                    rmtree(str(managed_elan_root), ignore_errors=True)
                env["_MOTO_LEAN4_REPAIR_ATTEMPTED"] = "1"
                install_lean4(runtime, env, _is_repair=True)
                return
            except Exception as repair_exc:
                cprint("WARNING: Lean 4 repair failed -- proof verification will be skipped.", YELLOW)
                cprint(str(repair_exc), YELLOW)
        else:
            cprint("WARNING: Lean 4 tooling still incomplete after repair -- proof verification will be skipped.", YELLOW)
        _set_lean_env_flags(env, enabled=False)
        print()
        return

    try:
        lean_version = subprocess.check_output([lean_cmd, "--version"], text=True).strip()
        cprint(f"Lean 4 ready: {lean_version}", GREEN)
    except Exception as exc:
        # The lean binary exists but is broken (corrupted toolchain, bad elan state,
        # incomplete update, etc.).  Wipe the elan directory and retry installation
        # once rather than giving up — this is the same non-fatal install path used
        # when Lean is missing entirely.
        cprint("Lean 4 version check failed — installation may be corrupt.", YELLOW)
        cprint(str(exc), YELLOW)
        if env.get("_MOTO_LEAN4_REPAIR_ATTEMPTED") != "1":
            cprint("Attempting to repair by wiping elan and reinstalling...", YELLOW)
            print()
            try:
                if elan_bin_dir.parent.exists():
                    rmtree(str(elan_bin_dir.parent), ignore_errors=True)
                managed_elan_root = Path(runtime.data_root) / "elan"
                if managed_elan_root.exists():
                    rmtree(str(managed_elan_root), ignore_errors=True)
                # Verify the wipe actually removed the binary — on Windows, file
                # locks from the just-invoked lean process can cause rmtree to
                # silently skip files when ignore_errors=True.
                if get_lean_command() is not None:
                    raise RuntimeError(
                        "Could not remove corrupt Lean binary (file may be locked). "
                        "Try closing any running Lean processes and relaunching MOTO."
                    )
                env["_MOTO_LEAN4_REPAIR_ATTEMPTED"] = "1"
                install_lean4(runtime, env, _is_repair=True)
                return
            except Exception as repair_exc:
                cprint("WARNING: Lean 4 repair failed -- proof verification will be skipped.", YELLOW)
                cprint(str(repair_exc), YELLOW)
        else:
            cprint("WARNING: Lean 4 still broken after repair -- proof verification will be skipped.", YELLOW)
        _set_lean_env_flags(env, enabled=False)
        print()
        return

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["git", "config", "--global", "core.longpaths", "true"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            cprint(f"WARNING: Could not enable Git long paths automatically: {exc}", YELLOW)

    workspace_dir = Path(runtime.data_root) / "lean4_workspace"
    try:
        _write_lean_workspace_files(workspace_dir)
    except Exception as exc:
        cprint("WARNING: Lean 4 workspace files could not be prepared; proof verification may fail later.", YELLOW)
        cprint(str(exc), YELLOW)

    _set_lean_env_flags(
        env,
        enabled=True,
        lean_path=str(Path(lean_cmd).resolve()),
        workspace_dir=str(workspace_dir.resolve()),
    )
    cprint(f"Lean 4 workspace: {workspace_dir}", GREEN)
    print()


def install_z3(runtime: InstanceRuntime, env: dict[str, str]) -> None:
    """
    Ensure Z3 is available for optional SMT workflows.

    This step is intentionally non-fatal: if installation fails, MOTO still
    launches and simply disables SMT-related runtime wiring.
    """
    cprint("[4d/8] Checking optional Z3 / SMT solver...", YELLOW)
    print()

    managed_root = Path(runtime.data_root) / "z3"
    z3_cmd = get_z3_command()

    if not z3_cmd and managed_root.exists():
        managed_binary = _find_z3_binary(managed_root)
        if managed_binary is not None:
            _prepend_path_entry(str(managed_binary.parent), env)
            z3_cmd = str(managed_binary.resolve())

    if not z3_cmd:
        cprint("Z3 not detected. Attempting one-time download...", YELLOW)
        try:
            release_request = Request(
                "https://api.github.com/repos/Z3Prover/z3/releases/latest",
                headers={
                    "User-Agent": "MOTO Launcher",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urlopen(release_request, timeout=60) as response:
                release_payload = json.loads(response.read().decode("utf-8"))

            asset = _select_z3_asset(list(release_payload.get("assets") or []))
            archive_path = managed_root / "downloads" / asset["name"]
            install_root = managed_root / "current"

            if archive_path.exists() and not _is_valid_archive(archive_path):
                cprint("Cached Z3 archive appears corrupt — re-downloading...", YELLOW)
                archive_path.unlink(missing_ok=True)

            if not archive_path.exists():
                _download_file(asset["browser_download_url"], archive_path)

            if not install_root.exists() or _find_z3_binary(install_root) is None:
                _extract_archive(archive_path, install_root)

            managed_binary = _find_z3_binary(install_root)
            if managed_binary is None:
                raise RuntimeError("Z3 archive extracted successfully, but the z3 binary could not be located.")

            if sys.platform != "win32":
                managed_binary.chmod(managed_binary.stat().st_mode | 0o111)

            _prepend_path_entry(str(managed_binary.parent), env)
            z3_cmd = str(managed_binary.resolve())
        except Exception as exc:
            print()
            cprint("WARNING: Z3 is not available -- SMT integration will remain disabled.", YELLOW)
            cprint(str(exc), YELLOW)
            _set_smt_env_flags(env, enabled=False)
            print()
            return

    try:
        z3_version = subprocess.check_output(
            [z3_cmd, "-version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        cprint(f"Z3 ready: {z3_version}", GREEN)
    except Exception as exc:
        cprint("WARNING: Z3 failed during version check -- SMT integration will remain disabled.", YELLOW)
        cprint(str(exc), YELLOW)
        _set_smt_env_flags(env, enabled=False)
        print()
        return

    _set_smt_env_flags(
        env,
        enabled=True,
        z3_path=str(Path(z3_cmd).resolve()),
    )
    cprint(f"Z3 binary: {z3_cmd}", GREEN)
    print()


def npm_output_reports_vulnerabilities(output: str) -> bool:
    normalized = (output or "").lower()
    return (
        "vulnerabilit" in normalized
        and (
            "npm audit fix" in normalized
            or "severity" in normalized
            or "to address all issues" in normalized
        )
    )


def install_frontend_dependencies() -> tuple[str, bool]:
    cprint("[5/8] Checking Node.js dependencies...", YELLOW)
    frontend_path = str(SCRIPT_DIR / "frontend")
    if not os.path.isdir(frontend_path):
        print()
        cprint("============================================================", RED)
        cprint("ERROR: frontend directory not found!", RED)
        cprint("============================================================", RED)
        print()
        cprint("Make sure you're running this from the project root directory.", YELLOW)
        exit_with_pause(1)

    npm_cmd = get_npm_command()
    if not npm_cmd:
        print()
        cprint("============================================================", RED)
        cprint("ERROR: npm is not available in PATH", RED)
        cprint("============================================================", RED)
        print()
        cprint("Reinstall Node.js from https://nodejs.org/ and ensure npm is included in PATH.", YELLOW)
        exit_with_pause(1)

    npm_result = subprocess.run(
        [npm_cmd, "install"],
        cwd=frontend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    npm_output = npm_result.stdout or ""
    print(npm_output, end="")

    if npm_result.returncode != 0:
        print()
        cprint("============================================================", RED)
        cprint("ERROR: Failed to install Node.js dependencies", RED)
        cprint("============================================================", RED)
        print()
        cprint("Please check:", YELLOW)
        cprint("- Internet connection is working", YELLOW)
        cprint("- package.json exists in frontend directory", YELLOW)
        exit_with_pause(1)

    # ──────────────────────────────────────────────────────────────────────
    # SECURITY-CRITICAL: Automatic npm vulnerability remediation.
    # PERMANENT LAUNCHER INVARIANT: Do not remove, disable, weaken, or bypass
    # this npm audit-fix system. If an agent accidentally removes or breaks it,
    # the agent must restore it immediately with no exceptions before continuing
    # any other launcher/updater work.
    # npm audit fix is the program's self-healing mechanism for dependency
    # CVEs. Removing it leaves the runtime exposed to known exploits with
    # no user-visible signal that remediation has stopped occurring.
    # ──────────────────────────────────────────────────────────────────────
    vulnerability_warning = npm_output_reports_vulnerabilities(npm_output)
    if vulnerability_warning:
        print()
        cprint("npm reported vulnerabilities — running `npm audit fix`...", YELLOW)
        fix_result = subprocess.run(
            [npm_cmd, "audit", "fix"],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        fix_output = (fix_result.stdout or "").strip()
        if fix_output:
            print(fix_output)

        still_vulnerable = npm_output_reports_vulnerabilities(fix_output)
        if fix_result.returncode != 0 or still_vulnerable:
            cprint("Standard fix insufficient — running `npm audit fix --force`...", YELLOW)
            force_result = subprocess.run(
                [npm_cmd, "audit", "fix", "--force"],
                cwd=frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            force_output = (force_result.stdout or "").strip()
            if force_output:
                print(force_output)
            if force_result.returncode == 0:
                cprint("npm audit fix --force completed.", GREEN)
                vulnerability_warning = False
            else:
                cprint("npm audit fix --force could not fully resolve all vulnerabilities.", YELLOW)
        else:
            cprint("npm audit fix completed.", GREEN)
            vulnerability_warning = False

    cprint("Node.js dependencies up to date", GREEN)
    print()
    return npm_cmd, vulnerability_warning


def check_lm_studio() -> None:
    cprint("[6/8] Checking LM Studio...", YELLOW)
    print()
    lm_available = False
    try:
        urlopen("http://127.0.0.1:1234/v1/models", timeout=3)
        lm_available = True
    except (URLError, OSError):
        lm_available = False

    if lm_available:
        cprint("LM Studio is running and responding!", GREEN)
    else:
        cprint("================================================================", CYAN)
        cprint("NOTE: LM Studio is not detected on http://127.0.0.1:1234", CYAN)
        cprint("================================================================", CYAN)
        print()
        cprint("This is OK! You have two options for AI models:", YELLOW)
        print()
        cprint("  Option 1: LM Studio (Local)", YELLOW)
        cprint("    - Download from: https://lmstudio.ai/", WHITE)
        cprint("    - Load a model and start the Local Server", WHITE)
        print()
        cprint("  Option 2: OpenRouter (Cloud API)", YELLOW)
        cprint("    - Get an API key from: https://openrouter.ai/", WHITE)
        cprint("    - Configure in Settings tab after launch", WHITE)
        print()
        cprint("The system will still start with isolated instance settings.", GREEN)
    print()


def check_secure_keyring() -> None:
    cprint("[6b/8] Checking secure credential storage...", YELLOW)
    try:
        keyring = importlib.import_module("keyring")
        backend = keyring.get_keyring()
        backend_name = f"{backend.__class__.__module__}.{backend.__class__.__name__}"
        if backend.__class__.__module__.startswith("keyring.backends.fail"):
            cprint("WARNING: No OS keyring backend is available right now.", YELLOW)
            if is_linux():
                cprint("Saved OpenRouter and Wolfram keys will not persist until a Secret Service compatible keyring is available in your desktop session.", YELLOW)
                cprint("Ubuntu users typically resolve this by enabling a desktop keyring such as `gnome-keyring`.", YELLOW)
            else:
                cprint("Saved provider keys will not persist until the OS keyring becomes available.", YELLOW)
        else:
            cprint(f"Keyring backend: {backend_name}", GREEN)
    except Exception as exc:
        cprint(f"WARNING: Could not inspect OS keyring availability: {exc}", YELLOW)
    print()


def verify_instance_ports(runtime: InstanceRuntime) -> None:
    cprint("[7/8] Final launch checks...", YELLOW)
    if runtime.is_default and not runtime.explicit_override:
        assert_runtime_lock_available(runtime.data_root)
    if port_in_use(runtime.backend_port):
        raise RuntimeError(f"Backend port {runtime.backend_port} became occupied before launch.")
    if port_in_use(runtime.frontend_port):
        raise RuntimeError(f"Frontend port {runtime.frontend_port} became occupied before launch.")
    cprint("Instance resources are ready.", GREEN)
    print()


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_services(
    runtime: InstanceRuntime,
    env: dict[str, str],
    frontend_url: str,
    backend_url: str,
    npm_cmd: str,
) -> tuple[LaunchedService, LaunchedService]:
    cprint("[8/8] Starting services...", YELLOW)
    print()
    cprint("================================================================", CYAN)
    cprint(f"  Starting MOTO instance '{runtime.instance_id}'", CYAN)
    cprint("================================================================", CYAN)
    print()
    cprint(f"Backend API will run on: {backend_url}", GREEN)
    cprint(f"Frontend UI will run on: {frontend_url}", GREEN)
    print()
    if is_linux():
        terminal = resolve_linux_terminal()
        if terminal is not None:
            cprint(f"Launcher will open separate Linux service terminals via {terminal[0]}.", GREEN)
        else:
            cprint("No desktop terminal emulator was detected. Services will run in the background and write logs under the active log root.", YELLOW)
            if not has_desktop_session():
                cprint("No DISPLAY/WAYLAND desktop session is active, so you may need to open the frontend URL manually.", YELLOW)
        print()
    cprint("Starting services automatically in 3 seconds...", YELLOW)
    time.sleep(3)
    print()

    backend_args = [
        get_python_command(),
        "-m",
        "uvicorn",
        "backend.api.main:app",
        "--host",
        runtime.backend_host,
        "--port",
        str(runtime.backend_port),
        "--no-access-log",
    ]
    backend_service = launch_service(
        title=f"MOTO Backend [{runtime.instance_id}]",
        service_slug="backend",
        args=backend_args,
        cwd=str(SCRIPT_DIR),
        env=env,
        log_root=runtime.log_root,
    )

    cprint("Waiting for backend to initialize...", YELLOW)
    time.sleep(5)
    if backend_service.mode != "window" and not is_pid_running(backend_service.pid):
        log_hint = f" Check {backend_service.log_path} for details." if backend_service.log_path else ""
        raise RuntimeError(f"{backend_service.title} exited during startup.{log_hint}")

    if runtime.is_default and not runtime.explicit_override:
        write_runtime_lock(runtime.data_root, backend_service.pid, runtime.instance_id, runtime.backend_port)

    frontend_service = launch_service(
        title=f"MOTO Frontend [{runtime.instance_id}]",
        service_slug="frontend",
        args=[npm_cmd, "run", "dev"],
        cwd=str(SCRIPT_DIR / "frontend"),
        env=env,
        log_root=runtime.log_root,
    )

    cprint("Waiting for frontend to initialize...", YELLOW)
    time.sleep(8)
    if frontend_service.mode != "window" and not is_pid_running(frontend_service.pid):
        log_hint = f" Check {frontend_service.log_path} for details." if frontend_service.log_path else ""
        raise RuntimeError(f"{frontend_service.title} exited during startup.{log_hint}")

    register_active_instance(
        instance_id=runtime.instance_id,
        backend_window_pid=backend_service.pid,
        frontend_window_pid=frontend_service.pid,
        backend_port=runtime.backend_port,
        frontend_port=runtime.frontend_port,
        data_root=runtime.data_root,
        log_root=runtime.log_root,
        keyring_namespace=runtime.secret_namespace,
        storage_prefix=runtime.storage_prefix,
    )

    # Persist the active instance runtime so subsequent relaunches can reuse
    # the same keyring namespace / data root / storage prefix. This includes
    # "default" launches — previously those were skipped, which caused the
    # keyring namespace to flip between None and a freshly minted timestamp
    # whenever the default ports happened to be busy between runs, and
    # therefore made saved OpenRouter / Wolfram Alpha keys look like they
    # "disappeared" on roughly every other launch.
    #
    # We intentionally SKIP this save for explicit overrides (MOTO_INSTANCE_ID
    # / MOTO_DATA_ROOT / MOTO_SECRET_NAMESPACE etc.). Those are one-off
    # overrides by design; persisting them would silently redirect the next
    # plain launch at the override's data root and keyring namespace, which
    # would look like the user's default instance had disappeared.
    if not runtime.explicit_override:
        try:
            save_last_instance_record(
                instance_id=runtime.instance_id,
                data_root=runtime.data_root,
                log_root=runtime.log_root,
                keyring_namespace=runtime.secret_namespace,
                storage_prefix=runtime.storage_prefix,
            )
        except OSError as exc:
            cprint(f"Warning: could not persist last-instance record: {exc}", YELLOW)

    cprint("Opening browser...", GREEN)
    webbrowser.open(frontend_url)
    return backend_service, frontend_service


def print_success_footer(
    runtime: InstanceRuntime,
    frontend_url: str,
    vulnerability_warning: bool,
    backend_service: LaunchedService,
    frontend_service: LaunchedService,
) -> None:
    print()
    cprint("================================================================", CYAN)
    cprint("  INSTANCE STARTED", CYAN)
    cprint("================================================================", CYAN)
    print()
    if backend_service.mode == "window" and frontend_service.mode == "window":
        cprint("Two service windows have opened:", GREEN)
    elif backend_service.mode == "terminal" and frontend_service.mode == "terminal":
        cprint("Two service terminals have opened:", GREEN)
    else:
        cprint("MOTO started the following launcher-managed services:", GREEN)
    cprint(f"  - MOTO Backend [{runtime.instance_id}] on port {runtime.backend_port}", GREEN)
    cprint(f"  - MOTO Frontend [{runtime.instance_id}] on port {runtime.frontend_port}", GREEN)
    if backend_service.log_path:
        cprint(f"    Backend log: {backend_service.log_path}", WHITE)
    if frontend_service.log_path:
        cprint(f"    Frontend log: {frontend_service.log_path}", WHITE)
    print()
    cprint("Browser opened automatically to:", GREEN)
    cprint(f"  {frontend_url}", CYAN)
    print()
    if vulnerability_warning:
        cprint("npm audit fix could not fully resolve all reported vulnerabilities. Review with `npm audit` in `frontend/`.", YELLOW)
        print()
    if backend_service.mode == "background" or frontend_service.mode == "background":
        cprint(f"To stop this instance: stop the launcher-managed backend/frontend processes for {runtime.instance_id}.", YELLOW)
    else:
        cprint(f"To stop this instance: close both service terminals/windows for {runtime.instance_id}.", YELLOW)
    print()
    cprint("This launcher window will close automatically.", GREEN)
    print()


def main() -> int:
    launcher_args, cleanup_targets = consume_internal_launcher_args(sys.argv[1:])
    _enable_ansi_on_windows()

    try:
        cleanup_relaunch_artifacts(cleanup_targets)
        clear_console()
        print_banner()

        if not handle_available_updates(launcher_args):
            return 0

        check_python_installation()
        check_node_installation()
        runtime, frontend_url, backend_url, env = prepare_runtime_and_environment()
        install_python_dependencies()
        install_playwright_browser()
        install_lean4(runtime, env)
        install_z3(runtime, env)
        npm_cmd, vulnerability_warning = install_frontend_dependencies()
        check_lm_studio()
        check_secure_keyring()
        verify_instance_ports(runtime)
        backend_service, frontend_service = start_services(runtime, env, frontend_url, backend_url, npm_cmd)
        print_success_footer(runtime, frontend_url, vulnerability_warning, backend_service, frontend_service)
        return 0
    except Exception as exc:
        print()
        cprint("============================================================", RED)
        cprint(f"FATAL ERROR: {exc}", RED)
        cprint("============================================================", RED)
        print()
        import traceback

        cprint("Stack Trace:", YELLOW)
        traceback.print_exc()
        exit_with_pause(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
