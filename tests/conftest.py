"""Session-wide safety rails for the MOTO test suite.

* Isolation: the whole session runs against a temporary data/log root, so no
  test can write into the developer's real ``backend/data`` or ``backend/logs``
  (a running desktop backend shares those roots).
* Write guard: any write that still reaches the real roots (for example a
  hard-coded path) fails the session and names the test that did it.
* Hang guard: a test that runs past ``MOTO_TEST_TIMEOUT_SECONDS`` (default 600;
  ``0`` disables) prints its node ID and every thread's stack, then ends the
  process instead of looping forever. Production workflows are intentionally unbounded, so a test
  that drives one without a stop condition would otherwise never finish.
"""
from __future__ import annotations

import faulthandler
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_ROOTS = tuple(
    os.path.normcase(str((REPO_ROOT / "backend" / name).resolve()))
    for name in ("data", "logs")
)
_DEFAULT_TEST_TIMEOUT_SECONDS = 600
_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
)

_current_test = "<session setup>"
_real_root_writes: dict[str, set[str]] = {}
_session_root: Path | None = None
_original_roots: tuple[str, str] | None = None


def _under_real_root(path: object) -> str:
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, (str, os.PathLike)):
        return ""
    resolved = os.path.normcase(os.path.abspath(os.fspath(path)))
    for root in _REAL_ROOTS:
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved
    return ""


def _is_write_open(mode: object, flags: object) -> bool:
    if isinstance(mode, str):
        return any(marker in mode for marker in ("w", "a", "x", "+"))
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _audit_real_root_writes(event: str, args: tuple) -> None:
    # Audit hooks must never raise into the code being audited.
    try:
        if event == "open":
            if len(args) >= 3 and _is_write_open(args[1], args[2]):
                target = _under_real_root(args[0])
            else:
                return
        elif event in {"os.rename", "os.replace"}:
            target = _under_real_root(args[1]) if len(args) > 1 else ""
        elif event in {"os.remove", "os.unlink", "os.mkdir", "os.rmdir", "shutil.rmtree"}:
            target = _under_real_root(args[0]) if args else ""
        elif event == "sqlite3.connect":
            target = _under_real_root(args[0]) if args else ""
        else:
            return
        if target:
            _real_root_writes.setdefault(target, set()).add(_current_test)
    except Exception:
        pass


def pytest_configure(config: pytest.Config) -> None:
    global _session_root, _original_roots
    from backend.shared.config import bind_runtime_roots, system_config

    sys.addaudithook(_audit_real_root_writes)
    # A harness that already isolated this process (for example a deep child run
    # with its own MOTO_DATA_ROOT) keeps its roots; only a session that would
    # otherwise use the real backend roots is rebound to a temporary one.
    if not any(
        _under_real_root(root) or not Path(root).is_absolute()
        for root in (system_config.data_dir, system_config.logs_dir)
        if root
    ):
        return
    _original_roots = (system_config.data_dir, system_config.logs_dir)
    _session_root = Path(tempfile.mkdtemp(prefix="moto-test-roots-"))
    bind_runtime_roots(_session_root / "data", _session_root / "logs")


def pytest_unconfigure(config: pytest.Config) -> None:
    from backend.shared.config import bind_runtime_roots

    if _original_roots is not None:
        bind_runtime_roots(*_original_roots)
    if _session_root is not None:
        shutil.rmtree(_session_root, ignore_errors=True)


@pytest.fixture(autouse=True, scope="session")
def _block_real_codex_websocket_transport():
    """Keep provider transports faked: Codex websocket turns are opt-in per test.

    The production default opens a real socket to the Codex backend. A websocket
    test sets ``client.websocket_transport_enabled = True`` and injects a fake
    ``_open_websocket`` on its own instance; a test that enables the transport
    without a fake fails here instead of reaching the network.
    """
    from backend.shared.openai_codex_client import OpenAICodexClient

    def _refuse_real_websocket(self, url, headers):
        raise AssertionError(
            "Codex websocket transport reached the real connector in tests; "
            "inject a fake _open_websocket on the client instance."
        )

    OpenAICodexClient.websocket_transport_enabled = False
    OpenAICodexClient._open_websocket = _refuse_real_websocket
    yield


def _test_timeout_seconds() -> int:
    raw = os.environ.get("MOTO_TEST_TIMEOUT_SECONDS", "").strip()
    try:
        return max(0, int(raw)) if raw else _DEFAULT_TEST_TIMEOUT_SECONDS
    except ValueError:
        return _DEFAULT_TEST_TIMEOUT_SECONDS


def _end_hung_test(config: pytest.Config, nodeid: str, timeout: int) -> None:
    """Report a hung test with every thread's stack, then end the process."""
    capture = config.pluginmanager.get_plugin("capturemanager")
    try:
        if capture is not None:
            capture.suspend_global_capture(in_=True)
    except Exception:
        pass
    try:
        sys.stderr.write(
            f"\n\nTest exceeded MOTO_TEST_TIMEOUT_SECONDS={timeout}: {nodeid}\n"
            "Thread stacks at timeout:\n"
        )
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
    finally:
        os._exit(1)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    global _current_test
    _current_test = item.nodeid
    timeout = _test_timeout_seconds()
    watchdog = None
    if timeout:
        watchdog = threading.Timer(
            timeout,
            _end_hung_test,
            args=(item.config, item.nodeid, timeout),
        )
        watchdog.daemon = True
        watchdog.start()
    try:
        yield
    finally:
        if watchdog is not None:
            watchdog.cancel()
        _current_test = "<between tests>"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _real_root_writes:
        return
    lines = [
        "Tests wrote into the real backend/data or backend/logs roots "
        "(use the session temp root, tmp_path, or a temporary system_config.data_dir):",
    ]
    for path in sorted(_real_root_writes):
        tests = ", ".join(sorted(_real_root_writes[path]))
        lines.append(f"  {path}  <-  {tests}")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "real runtime root writes", red=True)
        for line in lines:
            reporter.write_line(line)
    else:
        print("\n".join(lines), file=sys.stderr)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
