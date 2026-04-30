"""
MOTO System Launcher (Python)
This is an internal script. Use "Click To Launch MOTO.bat" on Windows or "Launch MOTO.sh" on Ubuntu 24.04.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
from random import randint
import re
import socket
import shlex
from shutil import which
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
    build_warning_message,
    check_for_updates,
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
    except Exception:
        return


def cprint(message: str, colour: str = RESET) -> None:
    print(f"{colour}{message}{RESET}")


def exit_with_pause(code: int = 0) -> None:
    print()
    cprint("Press Enter to close...", YELLOW)
    try:
        input()
    except EOFError:
        pass
    sys.exit(code)


def resolve_command(*names: str) -> str | None:
    for name in names:
        resolved = which(name)
        if resolved:
            return resolved
    return None


def command_exists(name: str) -> bool:
    return resolve_command(name) is not None


def get_python_command() -> str:
    return sys.executable or resolve_command("python3", "python") or "python"


def _path_is_within(root: Path, candidate: str | Path) -> bool:
    try:
        Path(candidate).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def using_repo_local_venv() -> bool:
    return _path_is_within(SCRIPT_DIR / ".venv", get_python_command())


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def shell_join(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def get_node_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("node.exe", "node")
    return resolve_command("node")


def get_npm_command() -> str | None:
    if sys.platform == "win32":
        return resolve_command("npm.cmd", "npm.exe", "npm")
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

    backend_host = os.environ.get("MOTO_BACKEND_HOST") or os.environ.get("HOST") or "0.0.0.0"

    explicit_backend_port = None
    for variable in ("MOTO_BACKEND_PORT", "PORT"):
        value = os.environ.get(variable)
        if value:
            explicit_backend_port = int(value)
            break

    explicit_frontend_port = None
    for variable in ("MOTO_FRONTEND_PORT", "FRONTEND_PORT"):
        value = os.environ.get(variable)
        if value:
            explicit_frontend_port = int(value)
            break

    default_data = str((SCRIPT_DIR / "backend" / "data").resolve())
    default_log = str((SCRIPT_DIR / "backend" / "logs").resolve())
    default_backend = 8000
    default_frontend = 5173

    has_explicit_runtime = any(
        value is not None
        for value in (
            explicit_id,
            explicit_data,
            explicit_log,
            explicit_backend_port,
            explicit_frontend_port,
            explicit_secret,
            explicit_storage,
        )
    )

    # ------------------------------------------------------------------
    # CRITICAL: keyring namespace stability across every relaunch.
    #
    # Previously, a "fresh" launch with free default ports would mint
    # `instance_id="default"` with `secret_namespace=None` but never record
    # that choice. If the very next launch happened to find default ports
    # busy (e.g. Windows TIME_WAIT on 8000/5173), the launcher fell back to
    # minting a brand-new timestamped instance_id and therefore a brand-new
    # keyring service name, which made the saved OpenRouter / Wolfram keys
    # look like they had disappeared. A third launch would flip back to the
    # default namespace and "rediscover" the original key. This is the
    # 1/3-startup key-loss symptom.
    #
    # Fix: `save_last_instance_record` is invoked for EVERY non-explicit
    # launch (including default), and here on every non-explicit relaunch we
    # prefer the recorded runtime when it is not currently live — regardless
    # of whether the default ports are free. This keeps the keyring service
    # name and data root perfectly stable across restarts.
    # ------------------------------------------------------------------
    reused_record: dict | None = None
    blocked_record_instance_id: str | None = None
    if not has_explicit_runtime:
        last_record = load_last_instance_record()
        if last_record is not None:
            candidate_id = sanitize_instance_id(last_record.get("instance_id")) or "default"
            live_instance_ids = {
                str(active.get("instance_id") or "").strip()
                for active in cleanup_launcher_state()
                if isinstance(active, dict)
            }
            if candidate_id not in live_instance_ids:
                reused_record = {
                    "instance_id": candidate_id,
                    "data_root": last_record.get("data_root") or None,
                    "log_root": last_record.get("log_root") or None,
                    "secret_namespace": last_record.get("secret_namespace"),
                    "storage_prefix": last_record.get("storage_prefix"),
                }
            else:
                blocked_record_instance_id = candidate_id

    # Decide the instance identity.
    if has_explicit_runtime:
        instance_id = explicit_id or new_instance_id()
    elif reused_record is not None:
        instance_id = reused_record["instance_id"]
    else:
        # Very first launch on this install (no recorded runtime yet).
        # Only "adopt" the default instance if the default ports are
        # currently free AND the recorded live instance is not already using
        # the default identity. Otherwise mint a fresh namespace so we do not
        # collide with an active default data root/keyring namespace, or with
        # whatever process is holding 8000/5173.
        defaults_free = not port_in_use(default_backend) and not port_in_use(default_frontend)
        instance_id = (
            "default"
            if defaults_free and blocked_record_instance_id != "default"
            else new_instance_id()
        )

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
        secret_namespace = explicit_secret or (reused_record or {}).get("secret_namespace")
        storage_prefix = explicit_storage or (reused_record or {}).get("storage_prefix")
    else:
        recorded_secret = (reused_record or {}).get("secret_namespace")
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
        explicit_override=has_explicit_runtime,
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
    if sys.platform != "win32" or not candidate or not os.path.isabs(candidate):
        return candidate

    executable_path = Path(candidate)
    command_name = executable_path.name
    resolved = resolve_command(command_name)
    if not resolved:
        return candidate

    try:
        if Path(resolved).resolve() == executable_path.resolve():
            return command_name
    except OSError:
        return candidate

    return candidate


def windows_service_requires_direct_launch(args: Sequence[str]) -> bool:
    """Return True when cmd.exe quoting would be unsafe for this command."""
    if sys.platform != "win32" or not args:
        return False

    executable = str(args[0] or "").strip()
    if not executable or not os.path.isabs(executable):
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
        cprint("ERROR: Python 3.8+ is required to run the launcher", RED)
        cprint("============================================================", RED)
        print()
        if is_linux():
            cprint("Install Python 3 and python3-venv, then launch via `Launch MOTO.sh`.", YELLOW)
            cprint("Example: sudo apt install python3 python3-venv", YELLOW)
        else:
            cprint("Please install Python 3.8+ from:", YELLOW)
            cprint("https://www.python.org/downloads/", YELLOW)
            print()
            cprint("IMPORTANT: Check 'Add Python to PATH' during installation", YELLOW)
        exit_with_pause(1)

    version = subprocess.check_output([python_cmd, "--version"], text=True).strip()
    cprint(version, GREEN)
    cprint(f"Interpreter: {python_cmd}", WHITE)
    if is_linux():
        if using_repo_local_venv():
            cprint("Using repo-local .venv for Ubuntu-safe package installs.", GREEN)
        else:
            cprint("Tip: `Launch MOTO.sh` is the recommended Ubuntu 24.04 entrypoint because it keeps Python packages inside the repo-local .venv.", YELLOW)
    print()


def check_node_installation() -> None:
    cprint("[2/8] Checking Node.js installation...", YELLOW)
    node_cmd = get_node_command()
    if not node_cmd:
        print()
        cprint("============================================================", RED)
        cprint("ERROR: Node.js is not installed or not in PATH", RED)
        cprint("============================================================", RED)
        print()
        if is_linux():
            cprint("Install Node.js 16+ from nodejs.org or your Ubuntu package source, then retry.", YELLOW)
        else:
            cprint("Please install Node.js 16+ from:", YELLOW)
            cprint("https://nodejs.org/", YELLOW)
        exit_with_pause(1)

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
    npm_version = subprocess.check_output([npm_cmd, "--version"], text=True).strip()
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
                "Reusing previously launched instance runtime (same secret namespace, same data root).",
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
        cprint(f"Secret namespace: {runtime.secret_namespace}", WHITE)
    else:
        cprint("Secret namespace: shared default store", WHITE)
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
            cprint("- On Ubuntu 24.04, prefer launching via `Launch MOTO.sh` so installs stay inside the repo-local .venv", YELLOW)
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
    current_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    try:
        normalized_entry = str(Path(path_entry).resolve())
    except OSError:
        normalized_entry = path_entry
    normalized_parts = set()
    for part in current_parts:
        try:
            normalized_parts.add(str(Path(part).resolve()))
        except OSError:
            normalized_parts.add(part)
    if normalized_entry not in normalized_parts:
        os.environ["PATH"] = normalized_entry + os.pathsep + os.environ.get("PATH", "")
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
    """Download a remote file to disk using the standard library."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "MOTO Launcher"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract a zip or tarball into the destination directory."""
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


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
    env["MOTO_LEAN4_PROOF_TIMEOUT"] = env.get("MOTO_LEAN4_PROOF_TIMEOUT", "").strip() or "120"
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


def install_lean4(runtime: InstanceRuntime, env: dict[str, str]) -> None:
    """
    Ensure Lean 4 / elan is available for proof verification.

    This step is intentionally non-fatal: if installation fails, MOTO still
    launches and simply skips automated proof verification.
    """
    cprint("[4c/8] Checking Lean 4 / elan for proof verification...", YELLOW)
    print()

    elan_bin_dir = Path.home() / ".elan" / "bin"
    lean_cmd = get_lean_command()
    lake_cmd = get_lake_command()

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
                        "curl https://elan.lean-lang.org/install.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable",
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
        cprint("WARNING: Lean 4 tooling is incomplete -- proof verification will be skipped.", YELLOW)
        _set_lean_env_flags(env, enabled=False)
        print()
        return

    try:
        lean_version = subprocess.check_output([lean_cmd, "--version"], text=True).strip()
        cprint(f"Lean 4 ready: {lean_version}", GREEN)
    except Exception as exc:
        cprint("WARNING: Lean 4 verification failed during version check -- proof verification will be skipped.", YELLOW)
        cprint(str(exc), YELLOW)
        _set_lean_env_flags(env, enabled=False)
        print()
        return

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

    vulnerability_warning = "vulnerabilities found" in npm_output.lower()
    if vulnerability_warning:
        print()
        cprint("NOTE: npm reported vulnerability warnings during install.", YELLOW)
        cprint("Build 1 no longer auto-runs `npm audit fix` because that can dirty a clean checkout and break updater eligibility.", YELLOW)
        cprint("If you want to mutate dependencies intentionally, run `npm audit fix` manually inside `frontend/`.", YELLOW)

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
        pass

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
        import keyring

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
        secret_namespace=runtime.secret_namespace,
        storage_prefix=runtime.storage_prefix,
    )

    # Persist the active instance runtime so subsequent relaunches can reuse
    # the same secret_namespace / data_root / storage_prefix. This includes
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
                secret_namespace=runtime.secret_namespace,
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
        cprint("npm install reported vulnerability warnings earlier. Build 1 leaves that decision manual so updater-safe checkouts stay clean.", YELLOW)
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
