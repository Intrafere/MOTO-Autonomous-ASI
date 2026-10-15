import io
import inspect
import os
from pathlib import Path
import tarfile
import tempfile
from unittest import TestCase, main, mock
import zipfile

from backend.shared.lean_workspace_paths import data_root_workspace_key
import moto_launcher


class BackendReadinessTests(TestCase):
    def setUp(self) -> None:
        self.service = moto_launcher.LaunchedService(
            title="MOTO Backend [test]",
            pid=123,
            mode="background",
            log_path="C:/logs/launcher_backend.log",
        )

    def test_health_poll_returns_healthy_payload(self) -> None:
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = b'{"status":"healthy","instance_id":"test"}'
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with mock.patch.object(moto_launcher, "is_pid_running", return_value=True):
            with mock.patch.object(moto_launcher, "urlopen", return_value=response):
                payload = moto_launcher.wait_for_backend_health(
                    "http://localhost:8000",
                    self.service,
                    timeout_seconds=1,
                    poll_interval_seconds=0,
                )
        self.assertEqual(payload["status"], "healthy")

    def test_early_exit_reports_backend_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "launcher_backend.log"
            log_path.write_text("startup header\nspecific backend traceback\n", encoding="utf-8")
            service = moto_launcher.LaunchedService(
                title=self.service.title,
                pid=self.service.pid,
                mode=self.service.mode,
                log_path=str(log_path),
            )
            with mock.patch.object(moto_launcher, "is_pid_running", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "specific backend traceback"):
                    moto_launcher.wait_for_backend_health(
                        "http://localhost:8000",
                        service,
                        timeout_seconds=1,
                        poll_interval_seconds=0,
                    )

    def test_frontend_early_exit_reports_frontend_log(self) -> None:
        frontend = moto_launcher.LaunchedService(
            title="MOTO Frontend [test]",
            pid=456,
            mode="background",
            log_path="C:/logs/launcher_frontend.log",
        )
        with mock.patch.object(moto_launcher, "is_pid_running", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "launcher_frontend.log"):
                moto_launcher.wait_for_frontend_ready(
                    "http://localhost:5173",
                    frontend,
                    timeout_seconds=1,
                    poll_interval_seconds=0,
                )


class ResolveInstanceRuntimeTests(TestCase):
    def setUp(self) -> None:
        # Runtime-selection unit tests must never inspect or terminate a real
        # user's backend. Lock ownership is covered in isolated OS tests.
        patcher = mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Resolve against a temporary install root: with the lease reported free,
        # runtime selection clears the default root's advisory lock, which must
        # never be the running backend's real backend/data/.moto_runtime.lock.
        install_root = tempfile.TemporaryDirectory()
        self.addCleanup(install_root.cleanup)
        root_patcher = mock.patch.object(moto_launcher, "SCRIPT_DIR", Path(install_root.name))
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def test_defaults_free_uses_default_instance(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                    runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertEqual(runtime.backend_port, 8000)
        self.assertEqual(runtime.frontend_port, 5173)
        self.assertTrue(runtime.is_default)
        self.assertFalse(runtime.explicit_override)
        self.assertIsNone(runtime.secret_namespace)
        self.assertTrue(runtime.data_root.endswith("backend\\data") or runtime.data_root.endswith("backend/data"))
        self.assertTrue(runtime.log_root.endswith("backend\\logs") or runtime.log_root.endswith("backend/logs"))

    def test_occupied_backend_keeps_default_memory_and_frontend_origin(self) -> None:
        def fake_port_in_use(port: int) -> bool:
            return port == 8000

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_test_1234"):
                            runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertEqual(runtime.backend_port, 8001)
        self.assertEqual(runtime.frontend_port, 5173)
        self.assertTrue(runtime.is_default)
        self.assertTrue(runtime.data_root.endswith("backend\\data") or runtime.data_root.endswith("backend/data"))
        self.assertTrue(runtime.log_root.endswith("backend\\logs") or runtime.log_root.endswith("backend/logs"))
        self.assertIsNone(runtime.secret_namespace)
        self.assertIsNone(runtime.storage_prefix)
        self.assertFalse(runtime.explicit_override)

    def test_occupied_default_frontend_blocks_plain_launch(self) -> None:
        def fake_port_in_use(port: int) -> bool:
            return port == 5173

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        with self.assertRaisesRegex(RuntimeError, "Frontend port 5173 is already in use"):
                            moto_launcher.resolve_instance_runtime()

    def test_port_only_override_does_not_create_isolated_identity(self) -> None:
        with mock.patch.dict(os.environ, {"MOTO_BACKEND_PORT": "8123"}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertEqual(runtime.backend_port, 8123)
        self.assertEqual(runtime.frontend_port, 5173)
        self.assertTrue(runtime.is_default)
        self.assertIsNone(runtime.secret_namespace)
        self.assertIsNone(runtime.storage_prefix)
        self.assertFalse(runtime.explicit_override)

    def test_frontend_port_only_override_is_rejected_for_default_identity(self) -> None:
        with mock.patch.dict(os.environ, {"MOTO_FRONTEND_PORT": "5174"}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_test_1234"):
                            with self.assertRaisesRegex(RuntimeError, "Frontend port overrides are disabled"):
                                moto_launcher.resolve_instance_runtime()

    def test_explicit_identity_frontend_port_override_is_allowed(self) -> None:
        with mock.patch.dict(os.environ, {"MOTO_INSTANCE_ID": "explicit_run", "MOTO_FRONTEND_PORT": "5174"}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None) as loader:
                with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                    runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "explicit_run")
        self.assertEqual(runtime.backend_port, 8000)
        self.assertEqual(runtime.frontend_port, 5174)
        self.assertFalse(runtime.is_default)
        self.assertTrue(runtime.explicit_override)
        loader.assert_not_called()

    def test_last_record_default_is_reused_when_backend_port_busy(self) -> None:
        """
        Regression test for the 1/3-startup keyring namespace drift bug.

        Previous behaviour: a fresh "default" launch never recorded itself,
        so if the second launch found the default ports busy (Windows
        TIME_WAIT is extremely common for this), the launcher would mint a
        brand-new timestamped instance_id with a brand-new keyring service
        name, and the OpenRouter/Wolfram keys would look like they had
        disappeared. Now a recorded "default" identity is reused even when
        the default ports are temporarily occupied — only the ports change.
        """
        def fake_port_in_use(port: int) -> bool:
            return port == 8000

        saved_record = {
            "instance_id": "default",
            "data_root": None,
            "log_root": None,
            "secret_namespace": None,
            "storage_prefix": None,
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertTrue(runtime.is_default)
        # The saved default namespace has None → keyring service name keeps
        # its legacy, suffix-free form so previously-saved keys stay visible.
        self.assertIsNone(runtime.secret_namespace)
        # Ports are allowed to shift because they are not part of the keyring
        # namespace — stability of `secret_namespace` is all that matters.
        self.assertNotEqual(runtime.backend_port, 8000)
        self.assertEqual(runtime.frontend_port, 5173)

    def test_last_record_default_blocks_when_frontend_origin_busy(self) -> None:
        saved_record = {
            "instance_id": "default",
            "data_root": None,
            "log_root": None,
            "secret_namespace": None,
            "storage_prefix": None,
        }

        def fake_port_in_use(port: int) -> bool:
            return port == 5173

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        with self.assertRaisesRegex(RuntimeError, "Frontend port 5173 is already in use"):
                            moto_launcher.resolve_instance_runtime()

    def test_stale_default_record_cannot_redirect_default_memory_or_storage(self) -> None:
        saved_record = {
            "instance_id": "default",
            "data_root": r"C:\\wrong\\data",
            "log_root": r"C:\\wrong\\logs",
            "keyring_namespace": "wrong_namespace",
            "storage_prefix": "wrong_storage",
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertTrue(runtime.is_default)
        self.assertTrue(runtime.data_root.endswith("backend\\data") or runtime.data_root.endswith("backend/data"))
        self.assertTrue(runtime.log_root.endswith("backend\\logs") or runtime.log_root.endswith("backend/logs"))
        self.assertIsNone(runtime.secret_namespace)
        self.assertIsNone(runtime.storage_prefix)

    def test_plain_launch_ignores_recorded_isolated_instance(self) -> None:
        """A plain consumer relaunch must return to the shared default memory/keyring."""
        saved_record = {
            "instance_id": "instance_20260101_000000_1111",
            "data_root": r"C:\\custom\\data",
            "log_root": r"C:\\custom\\logs",
            "secret_namespace": "instance_20260101_000000_1111",
            "storage_prefix": "instance_20260101_000000_1111",
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertTrue(runtime.is_default)
        self.assertIsNone(runtime.secret_namespace)
        self.assertIsNone(runtime.storage_prefix)
        self.assertTrue(runtime.data_root.endswith("backend\\data") or runtime.data_root.endswith("backend/data"))

    def test_live_non_default_record_does_not_create_new_plain_launch_namespace(self) -> None:
        """A recorded isolated instance never redirects a plain launch away from default."""
        saved_record = {
            "instance_id": "instance_20260101_000000_1111",
            "data_root": None,
            "log_root": None,
            "secret_namespace": "instance_20260101_000000_1111",
            "storage_prefix": "instance_20260101_000000_1111",
        }
        live_record = [{"instance_id": "instance_20260101_000000_1111"}]

        def fake_port_in_use(port: int) -> bool:
            return port == 8000

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=live_record):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_test_freshly_minted"):
                            runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "default")
        self.assertTrue(runtime.is_default)
        self.assertIsNone(runtime.secret_namespace)
        self.assertIsNone(runtime.storage_prefix)

    def test_live_default_instance_blocks_plain_relaunch(self) -> None:
        """A live default instance must not cause a new empty namespace."""
        saved_record = {
            "instance_id": "default",
            "data_root": None,
            "log_root": None,
            "secret_namespace": None,
            "storage_prefix": None,
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(
                    moto_launcher,
                    "assert_runtime_lock_available",
                    side_effect=RuntimeError("default MOTO instance is already running"),
                ):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        with self.assertRaisesRegex(RuntimeError, "default MOTO instance is already running"):
                            moto_launcher.resolve_instance_runtime()

    def test_explicit_override_does_not_read_last_record(self) -> None:
        """Explicit env overrides must never be replaced by a stored record."""
        saved_record = {
            "instance_id": "default",
            "data_root": None,
            "log_root": None,
            "secret_namespace": None,
            "storage_prefix": None,
        }
        with mock.patch.dict(os.environ, {"MOTO_INSTANCE_ID": "explicit_run"}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record) as loader:
                with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                    runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "explicit_run")
        self.assertTrue(runtime.explicit_override)
        self.assertEqual(runtime.secret_namespace, "explicit_run")
        # We must not even consult the stored last-instance record when the
        # caller provided explicit overrides.
        loader.assert_not_called()

    def test_explicit_secret_namespace_is_preserved(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"MOTO_INSTANCE_ID": "explicit_run", "MOTO_SECRET_NAMESPACE": "stored_keyring_namespace"},
            clear=True,
        ):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None) as loader:
                with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                    runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "explicit_run")
        self.assertEqual(runtime.secret_namespace, "stored_keyring_namespace")
        self.assertTrue(runtime.explicit_override)
        loader.assert_not_called()

    def test_orphan_recovery_happens_before_backend_port_selection(self) -> None:
        observed_ports: list[int] = []

        def fake_port_in_use(port: int) -> bool:
            observed_ports.append(port)
            return False

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                    with mock.patch.object(moto_launcher, "assert_runtime_lock_available") as reconcile:
                        with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                            runtime = moto_launcher.resolve_instance_runtime()

        reconcile.assert_called_once_with(runtime.data_root)
        self.assertEqual(runtime.backend_port, 8000)
        self.assertEqual(observed_ports[0], 8000)

    def test_tracked_backend_orphan_reaches_reconciliation(self) -> None:
        record = {
            "instance_id": "default",
            "backend_window_pid": 15216,
            "frontend_window_pid": 40124,
        }
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None), \
             mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[record]), \
             mock.patch.object(moto_launcher, "assert_runtime_lock_available") as reconcile, \
             mock.patch.object(moto_launcher, "port_in_use", return_value=False):
            runtime = moto_launcher.resolve_instance_runtime()
        reconcile.assert_called_once_with(runtime.data_root)

    def test_runtime_lock_ignores_stale_default_backend_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 4242, "default")
            with mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=False):
                moto_launcher.assert_runtime_lock_available(temp_dir)

    def test_runtime_lock_rejects_unverified_windows_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 37088, "default", 8000)
            with mock.patch.object(moto_launcher.sys, "platform", "win32"):
                with mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=True):
                    with mock.patch.object(moto_launcher, "read_backend_lease_owner", return_value=15216):
                        with mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8000]):
                            with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                                with mock.patch.object(moto_launcher, "windows_parent_pid", return_value=999):
                                    with mock.patch.object(moto_launcher, "backend_byte_zero_is_held", return_value=False):
                                        with mock.patch.object(moto_launcher, "is_pid_running", return_value=True):
                                            with mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True):
                                                with mock.patch.object(moto_launcher, "backend_health_identity", return_value="other"):
                                                    with mock.patch.object(moto_launcher, "terminate_process_tree") as terminate:
                                                        with self.assertRaisesRegex(RuntimeError, "could not safely identify"):
                                                            moto_launcher.assert_runtime_lock_available(temp_dir)
            terminate.assert_not_called()

    def test_runtime_lock_recovers_healthy_backend_from_recorded_port(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 37088, "default", 8000)
            with mock.patch.object(moto_launcher.sys, "platform", "win32"):
                with mock.patch.object(
                    moto_launcher,
                    "backend_lease_is_held",
                    side_effect=[True, False, False],
                ):
                    with mock.patch.object(
                        moto_launcher,
                        "port_in_use",
                        return_value=False,
                    ):
                        with mock.patch.object(moto_launcher, "read_backend_lease_owner", return_value=15216):
                            with mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8000]):
                                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]):
                                    with mock.patch.object(moto_launcher, "windows_parent_pid", return_value=37088):
                                        with mock.patch.object(moto_launcher, "backend_byte_zero_is_held", return_value=False):
                                            with mock.patch.object(
                                                moto_launcher,
                                                "backend_health_identity",
                                                return_value="default",
                                            ):
                                                with mock.patch.object(
                                                    moto_launcher,
                                                    "is_pid_running",
                                                    side_effect=lambda pid: pid == 15216,
                                                ):
                                                    with mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True):
                                                        with mock.patch.object(moto_launcher, "terminate_process_tree") as terminate:
                                                            moto_launcher.assert_runtime_lock_available(temp_dir)

            terminate.assert_called_once_with(15216)

    def test_runtime_lock_preserves_untracked_healthy_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 15216, "default", 8123)
            with mock.patch.object(moto_launcher.sys, "platform", "win32"), \
                 mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=True), \
                 mock.patch.object(moto_launcher, "read_backend_lease_owner", return_value=15216), \
                 mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8123]), \
                 mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]), \
                 mock.patch.object(moto_launcher, "windows_parent_pid", return_value=400), \
                 mock.patch.object(moto_launcher, "backend_byte_zero_is_held", return_value=False), \
                 mock.patch.object(moto_launcher, "is_pid_running", return_value=True), \
                 mock.patch.object(moto_launcher, "backend_health_identity", return_value="default"), \
                 mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True), \
                 mock.patch.object(moto_launcher, "terminate_process_tree") as terminate:
                with self.assertRaisesRegex(RuntimeError, "could not safely identify"):
                    moto_launcher.assert_runtime_lock_available(temp_dir)
            terminate.assert_not_called()

    def test_runtime_lock_preserves_modern_untracked_backend_with_dead_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 15216, "default", 8123)
            running = lambda pid: pid == 15216
            with mock.patch.object(moto_launcher.sys, "platform", "win32"), \
                 mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=True), \
                 mock.patch.object(moto_launcher, "read_backend_lease_owner", return_value=15216), \
                 mock.patch.object(moto_launcher, "backend_byte_zero_is_held", return_value=True), \
                 mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8123]), \
                 mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]), \
                 mock.patch.object(moto_launcher, "windows_parent_pid", return_value=400), \
                 mock.patch.object(moto_launcher, "is_pid_running", side_effect=running), \
                 mock.patch.object(moto_launcher, "backend_health_identity", return_value="default"), \
                 mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True), \
                 mock.patch.object(moto_launcher, "terminate_process_tree") as terminate:
                with self.assertRaisesRegex(RuntimeError, "could not safely identify"):
                    moto_launcher.assert_runtime_lock_available(temp_dir)
            terminate.assert_not_called()

    def test_runtime_lock_preserves_tracked_backend_when_frontend_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 15216, "default", 8123)
            record = {
                "instance_id": "default",
                "backend_window_pid": 15216,
                "frontend_window_pid": 40124,
                "data_root": temp_dir,
            }
            running = lambda pid: pid == 15216
            with mock.patch.object(moto_launcher.sys, "platform", "win32"), \
                 mock.patch.object(moto_launcher, "backend_lease_is_held", side_effect=[True, False, False]), \
                 mock.patch.object(moto_launcher, "read_backend_lease_owner", return_value=None), \
                 mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8123]), \
                 mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[record]), \
                 mock.patch.object(moto_launcher, "windows_parent_pid", return_value=1), \
                 mock.patch.object(moto_launcher, "is_pid_running", side_effect=running), \
                 mock.patch.object(moto_launcher, "backend_health_identity", return_value="default"), \
                 mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True), \
                 mock.patch.object(moto_launcher, "port_in_use", return_value=False), \
                 mock.patch.object(moto_launcher, "terminate_process_tree") as terminate:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    moto_launcher.assert_runtime_lock_available(temp_dir)
            terminate.assert_not_called()

    def test_pid_running_treats_windows_invalid_parameter_as_not_running(self) -> None:
        kernel32 = mock.MagicMock()
        kernel32.OpenProcess.return_value = 0
        kernel32.GetLastError.return_value = 87
        ctypes_module = mock.MagicMock()
        ctypes_module.windll.kernel32 = kernel32
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.dict("sys.modules", {"ctypes": ctypes_module}):
                self.assertFalse(moto_launcher.is_pid_running(4242))

    def test_pid_running_uses_windows_process_handle_for_live_process(self) -> None:
        kernel32 = mock.MagicMock()
        kernel32.OpenProcess.return_value = 99
        kernel32.GetExitCodeProcess.side_effect = lambda _handle, pointer: (
            setattr(pointer._obj, "value", 259) or 1
        )
        ctypes_module = mock.MagicMock()
        ctypes_module.windll.kernel32 = kernel32
        ctypes_module.byref.side_effect = lambda value: mock.Mock(_obj=value)
        wintypes_module = mock.MagicMock()

        class FakeDword:
            value = 0

        wintypes_module.DWORD = FakeDword
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.dict(
                "sys.modules",
                {"ctypes": ctypes_module, "ctypes.wintypes": wintypes_module},
            ):
                self.assertTrue(moto_launcher.is_pid_running(4242))

        kernel32.CloseHandle.assert_called_once_with(99)


class WindowsLauncherStrategyTests(TestCase):
    def test_build_windows_service_command_prefers_path_safe_executable_name(self) -> None:
        npm_path = r"C:\Program Files\nodejs\npm.cmd"

        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value=npm_path):
                command = moto_launcher.build_windows_service_command(
                    "MOTO Frontend [default]",
                    [npm_path, "run", "dev"],
                )

        self.assertIn("npm.cmd run dev", command)
        self.assertNotIn(npm_path, command)

    def test_launch_windows_service_falls_back_to_direct_launch_for_unsafe_absolute_path(self) -> None:
        tool_path = r"C:\Program Files\Custom Tools\frontend.cmd"
        process = mock.Mock(pid=5150)

        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value=None):
                with mock.patch.object(moto_launcher.subprocess, "Popen", return_value=process) as popen:
                    service = moto_launcher.launch_windows_service(
                        "MOTO Frontend [default]",
                        [tool_path, "run", "dev"],
                        cwd=r"C:\repo",
                        env={},
                    )

        self.assertEqual(service.mode, "window")
        self.assertEqual(service.pid, 5150)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], [tool_path, "run", "dev"])

    def test_launch_service_starts_windows_backend_directly(self) -> None:
        process = mock.Mock(pid=6160)
        args = [r"C:\Python\python.exe", "-m", "uvicorn", "backend.api.main:app"]

        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher.subprocess, "Popen", return_value=process) as popen:
                service = moto_launcher.launch_service(
                    "MOTO Backend [default]",
                    "backend",
                    args,
                    cwd=r"C:\repo",
                    env={},
                    log_root=r"C:\repo\backend\logs",
                )

        self.assertEqual(service.pid, 6160)
        launched_args = popen.call_args.args[0]
        self.assertEqual(launched_args[: len(args)], args)
        self.assertEqual(launched_args[-2], "--log-config")
        self.assertNotIn("stdout", popen.call_args.kwargs)
        self.assertNotIn("stderr", popen.call_args.kwargs)
        self.assertTrue(service.log_path.endswith("launcher_backend.log"))
        self.assertEqual(
            popen.call_args.kwargs["env"]["MOTO_BACKEND_CONSOLE_TITLE"],
            "MOTO Backend [default]",
        )


class ServiceStartupRollbackTests(TestCase):
    def _default_runtime(self) -> moto_launcher.InstanceRuntime:
        return moto_launcher.InstanceRuntime(
            instance_id="default",
            backend_host="127.0.0.1",
            backend_port=8000,
            frontend_port=5173,
            data_root=r"C:\data",
            log_root=r"C:\logs",
            secret_namespace=None,
            storage_prefix=None,
            is_default=True,
            explicit_override=False,
        )

    def test_failed_termination_preserves_runtime_owner_metadata(self) -> None:
        backend = moto_launcher.LaunchedService("backend", 101, "window")
        with mock.patch.object(moto_launcher, "launch_service", return_value=backend), \
             mock.patch.object(
                 moto_launcher,
                 "wait_for_backend_health",
                 side_effect=RuntimeError("startup failed"),
             ), \
             mock.patch.object(moto_launcher, "terminate_launched_service"), \
             mock.patch.object(moto_launcher, "is_pid_running", return_value=True), \
             mock.patch.object(moto_launcher, "backend_lease_is_held", return_value=True), \
             mock.patch.object(moto_launcher.time, "sleep"), \
             mock.patch.object(
                 moto_launcher.time,
                 "monotonic",
                 side_effect=[0.0, 0.0, 11.0],
             ), \
             mock.patch.object(moto_launcher, "remove_owned_runtime_lock") as remove:
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                moto_launcher.start_services(
                    self._default_runtime(),
                    {},
                    "http://127.0.0.1:5173",
                    "http://127.0.0.1:8000",
                    "npm",
                )
        remove.assert_not_called()

    def test_registration_failure_terminates_both_started_services(self) -> None:
        runtime = moto_launcher.InstanceRuntime(
            instance_id="test",
            backend_host="127.0.0.1",
            backend_port=8100,
            frontend_port=5174,
            data_root=r"C:\data",
            log_root=r"C:\logs",
            secret_namespace="test",
            storage_prefix="test",
            is_default=False,
            explicit_override=True,
        )
        backend = moto_launcher.LaunchedService("backend", 101, "window")
        frontend = moto_launcher.LaunchedService("frontend", 202, "window")

        with mock.patch.object(moto_launcher, "launch_service", side_effect=[backend, frontend]):
            with mock.patch.object(moto_launcher, "wait_for_backend_health"):
                with mock.patch.object(moto_launcher, "wait_for_frontend_ready"):
                    with mock.patch.object(
                        moto_launcher,
                        "register_active_instance",
                        side_effect=RuntimeError("registration failed"),
                    ):
                        with mock.patch.object(moto_launcher, "terminate_launched_service") as terminate:
                            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                                moto_launcher.start_services(
                                    runtime,
                                    {},
                                    "http://127.0.0.1:5174",
                                    "http://127.0.0.1:8100",
                                    "npm",
                                )

        self.assertEqual(
            terminate.call_args_list,
            [mock.call(frontend), mock.call(backend)],
        )


class LauncherDependencyVersionTests(TestCase):
    def test_node_version_support_matches_vite_engine_floor(self) -> None:
        self.assertFalse(moto_launcher.node_version_is_supported((20, 18, 1)))
        self.assertTrue(moto_launcher.node_version_is_supported((20, 19, 0)))
        self.assertFalse(moto_launcher.node_version_is_supported((21, 7, 0)))
        self.assertFalse(moto_launcher.node_version_is_supported((22, 11, 0)))
        self.assertTrue(moto_launcher.node_version_is_supported((22, 12, 0)))
        self.assertTrue(moto_launcher.node_version_is_supported((24, 0, 0)))

    def test_check_node_installation_uses_winget_when_missing_on_windows(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "get_node_command", side_effect=[None, r"C:\Program Files\nodejs\node.exe"]):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value=r"C:\Program Files\nodejs\npm.cmd"):
                    with mock.patch.object(moto_launcher, "install_windows_nodejs", return_value=True) as installer:
                        with mock.patch.object(moto_launcher.subprocess, "check_output", side_effect=["v22.12.0", "10.9.0"]):
                            moto_launcher.check_node_installation()

        installer.assert_called_once()

    def test_windows_truststore_bootstrap_skips_install_when_available(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "get_python_command", return_value="python"):
                with mock.patch.object(moto_launcher, "run_silent", return_value=0) as probe:
                    with mock.patch.object(moto_launcher, "run_visible") as install:
                        self.assertTrue(moto_launcher.ensure_windows_launcher_truststore())

        probe.assert_called_once_with(
            ["python", "-c", "import truststore"],
            cwd=str(moto_launcher.SCRIPT_DIR),
        )
        install.assert_not_called()

    def test_windows_truststore_bootstrap_installs_and_reprobes(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "get_python_command", return_value="python"):
                with mock.patch.object(
                    moto_launcher,
                    "run_silent",
                    side_effect=[1, 0],
                ) as probe:
                    with mock.patch.object(
                        moto_launcher,
                        "run_visible",
                        return_value=0,
                    ) as install:
                        with mock.patch.object(
                            moto_launcher.importlib,
                            "invalidate_caches",
                        ) as invalidate:
                            self.assertTrue(moto_launcher.ensure_windows_launcher_truststore())

        self.assertEqual(probe.call_count, 2)
        install.assert_called_once_with(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                moto_launcher.TRUSTSTORE_REQUIREMENT,
            ],
            cwd=str(moto_launcher.SCRIPT_DIR),
            check=False,
        )
        invalidate.assert_called_once_with()

    def test_windows_truststore_bootstrap_failure_keeps_startup_available(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "get_python_command", return_value="python"):
                with mock.patch.object(moto_launcher, "run_silent", return_value=1):
                    with mock.patch.object(moto_launcher, "run_visible", return_value=1):
                        self.assertFalse(moto_launcher.ensure_windows_launcher_truststore())

    def test_truststore_bootstrap_is_noop_outside_windows(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "linux"):
            with mock.patch.object(moto_launcher, "run_silent") as probe:
                self.assertTrue(moto_launcher.ensure_windows_launcher_truststore())

        probe.assert_not_called()

    def test_main_bootstraps_native_trust_before_update_check(self) -> None:
        source = inspect.getsource(moto_launcher.main)
        self.assertLess(
            source.index("ensure_windows_launcher_truststore()"),
            source.index("handle_available_updates(launcher_args)"),
        )

    def test_install_windows_nodejs_tries_user_scope_lts_after_source_refresh(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value="winget"):
                with mock.patch.object(moto_launcher, "run_visible", side_effect=[0, 0]) as run_visible:
                    self.assertTrue(moto_launcher.install_windows_nodejs())

        self.assertEqual(run_visible.call_args_list[0].args[0], ["winget", "source", "update", "--name", "winget"])
        self.assertEqual(
            run_visible.call_args_list[1].args[0],
            [
                "winget",
                "install",
                "--id",
                "OpenJS.NodeJS.LTS",
                "-e",
                "--source",
                "winget",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--scope",
                "user",
            ],
        )

    def test_install_windows_nodejs_falls_back_to_lts_default_then_non_lts_user_scope(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value="winget"):
                with mock.patch.object(moto_launcher, "run_visible", side_effect=[0, 1, 1, 0]) as run_visible:
                    self.assertTrue(moto_launcher.install_windows_nodejs())

        self.assertEqual(
            [call.args[0] for call in run_visible.call_args_list],
            [
                ["winget", "source", "update", "--name", "winget"],
                [
                    "winget",
                    "install",
                    "--id",
                    "OpenJS.NodeJS.LTS",
                    "-e",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "--scope",
                    "user",
                ],
                [
                    "winget",
                    "install",
                    "--id",
                    "OpenJS.NodeJS.LTS",
                    "-e",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                [
                    "winget",
                    "install",
                    "--id",
                    "OpenJS.NodeJS",
                    "-e",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "--scope",
                    "user",
                ],
            ],
        )

    def test_install_windows_git_tries_user_scope_then_default_scope(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value="winget"):
                with mock.patch.object(moto_launcher, "run_visible", return_value=0) as run_visible:
                    with mock.patch.object(moto_launcher, "refresh_windows_path_from_registry"):
                        with mock.patch.object(
                            moto_launcher,
                            "_verified_git_candidate",
                            side_effect=[
                                (None, "not found after user install"),
                                (r"C:\Program Files\Git\cmd\git.exe", "git version 2.51.0"),
                            ],
                        ):
                            git_cmd, detail = moto_launcher.install_windows_git()

        self.assertEqual(git_cmd, r"C:\Program Files\Git\cmd\git.exe")
        self.assertEqual(detail, "git version 2.51.0")
        self.assertEqual(
            [call.args[0] for call in run_visible.call_args_list],
            [
                ["winget", "source", "update", "--name", "winget"],
                [
                    "winget",
                    "install",
                    "--id",
                    "Git.Git",
                    "-e",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "--scope",
                    "user",
                ],
                [
                    "winget",
                    "install",
                    "--id",
                    "Git.Git",
                    "-e",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
            ],
        )

    def test_ensure_git_for_lean_uses_existing_verified_git_without_install(self) -> None:
        env = {"PATH": os.environ.get("PATH", "")}
        probe = mock.Mock(returncode=0, stdout="git version 2.51.0\n", stderr="")
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "get_git_command", return_value=r"C:\Git\cmd\git.exe"):
                with mock.patch.object(moto_launcher.subprocess, "run", return_value=probe):
                    with mock.patch.object(moto_launcher, "install_windows_git") as installer:
                        with mock.patch.object(moto_launcher, "_prepend_path_entry") as prepend:
                            git_cmd = moto_launcher.ensure_git_for_lean(env)

        self.assertEqual(git_cmd, str(Path(r"C:\Git\cmd\git.exe").resolve()))
        installer.assert_not_called()
        prepend.assert_called_once_with(str(Path(git_cmd).parent), env)

    def test_ensure_git_for_lean_propagates_verified_git_directory_to_child_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            git_path = Path(temp_dir) / "git.exe"
            git_path.write_text("", encoding="utf-8")
            env = {"PATH": r"C:\Windows\System32"}
            probe = mock.Mock(returncode=0, stdout="git version 2.51.0\n", stderr="")

            with mock.patch.object(moto_launcher.sys, "platform", "win32"):
                with mock.patch.object(moto_launcher, "get_git_command", return_value=str(git_path)):
                    with mock.patch.object(moto_launcher.subprocess, "run", return_value=probe):
                        with mock.patch.dict(os.environ, {"PATH": env["PATH"]}, clear=False):
                            resolved_git = moto_launcher.ensure_git_for_lean(env)

            self.assertEqual(resolved_git, str(git_path.resolve()))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(git_path.parent.resolve()))

    def test_ensure_git_for_lean_reprobes_after_windows_install_and_path_refresh(self) -> None:
        env = {"PATH": ""}
        installed_git = r"C:\Program Files\Git\cmd\git.exe"
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_verified_git_candidate",
                return_value=(None, "broken"),
            ):
                with mock.patch.object(
                    moto_launcher,
                    "install_windows_git",
                    return_value=(installed_git, "git version 2.51.0"),
                ) as installer:
                    with mock.patch.object(moto_launcher, "_prepend_path_entry") as prepend:
                        git_cmd = moto_launcher.ensure_git_for_lean(env)

        self.assertEqual(git_cmd, installed_git)
        installer.assert_called_once_with()
        prepend.assert_called_once_with(str(Path(installed_git).parent), env)

    def test_ensure_git_for_lean_does_not_trust_successful_winget_without_probe(self) -> None:
        env = {"PATH": ""}
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_verified_git_candidate",
                return_value=(None, "not found"),
            ):
                with mock.patch.object(
                    moto_launcher,
                    "install_windows_git",
                    return_value=(None, "not found after installation"),
                ):
                    self.assertIsNone(moto_launcher.ensure_git_for_lean(env))

    def test_install_lean4_disables_proofs_when_git_is_unavailable(self) -> None:
        runtime = mock.Mock(data_root=r"C:\MOTO\data")
        env: dict[str, str] = {}
        with mock.patch.object(moto_launcher, "ensure_git_for_lean", return_value=None):
            with mock.patch.object(moto_launcher, "_set_lean_env_flags") as set_flags:
                with mock.patch.object(moto_launcher, "get_lean_command") as get_lean:
                    moto_launcher.install_lean4(runtime, env)

        set_flags.assert_called_once_with(
            env,
            enabled=False,
            unavailable_reason="git_unavailable",
        )
        get_lean.assert_not_called()

    def test_ensure_git_for_lean_linux_missing_git_skips_windows_installer(self) -> None:
        env = {"PATH": ""}
        with mock.patch.object(moto_launcher.sys, "platform", "linux"):
            with mock.patch.object(moto_launcher, "get_git_command", return_value=None):
                with mock.patch.object(moto_launcher, "install_windows_git") as installer:
                    self.assertIsNone(moto_launcher.ensure_git_for_lean(env))

        installer.assert_not_called()

    def test_install_lean4_longpaths_uses_resolved_git(self) -> None:
        runtime = mock.Mock(data_root=r"C:\MOTO\data")
        env: dict[str, str] = {}
        git_cmd = r"C:\Program Files\Git\cmd\git.exe"
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "ensure_git_for_lean", return_value=git_cmd):
                with mock.patch.object(moto_launcher, "get_lean_command", return_value=r"C:\elan\lean.exe"):
                    with mock.patch.object(moto_launcher, "get_lake_command", return_value=r"C:\elan\lake.exe"):
                        with mock.patch.object(
                            moto_launcher.subprocess,
                            "check_output",
                            return_value="Lean (version 4.24.0)",
                        ):
                            with mock.patch.object(moto_launcher.subprocess, "run") as run:
                                with mock.patch.object(moto_launcher, "_write_lean_workspace_files"):
                                    with mock.patch.object(moto_launcher, "_set_lean_env_flags"):
                                        moto_launcher.install_lean4(runtime, env)

        run.assert_called_once_with(
            [git_cmd, "config", "--global", "core.longpaths", "true"],
            check=False,
            stdout=moto_launcher.subprocess.DEVNULL,
            stderr=moto_launcher.subprocess.DEVNULL,
        )

    def _install_lean4_on_windows(self, runtime, env):
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "ensure_git_for_lean", return_value="git"):
                with mock.patch.object(moto_launcher, "get_lean_command", return_value=r"C:\elan\lean.exe"):
                    with mock.patch.object(moto_launcher, "get_lake_command", return_value=r"C:\elan\lake.exe"):
                        with mock.patch.object(
                            moto_launcher.subprocess,
                            "check_output",
                            return_value="Lean (version 4.24.0)",
                        ):
                            with mock.patch.object(moto_launcher.subprocess, "run"):
                                with mock.patch.object(
                                    moto_launcher, "_write_lean_workspace_files"
                                ) as write_files:
                                    with mock.patch.object(moto_launcher, "_set_lean_env_flags") as set_flags:
                                        moto_launcher.install_lean4(runtime, env)
        return write_files, set_flags

    def test_install_lean4_moves_long_workspace_to_short_folder_and_removes_old_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / ("deep" * 15) / "backend" / "data"
            old_workspace = data_root / "lean4_workspace"
            (old_workspace / ".lake" / "packages").mkdir(parents=True)
            (old_workspace / "MOTOProofWorkspace.lean").write_text("import Mathlib\n", encoding="utf-8")
            (old_workspace / ".lake" / "packages" / "partial.olean").write_bytes(b"x" * 1024)
            local_app_data = root / "appdata"
            runtime = mock.Mock(data_root=str(data_root))
            env: dict[str, str] = {}

            with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}):
                write_files, set_flags = self._install_lean4_on_windows(runtime, env)

            relocated = local_app_data / "MOTO" / "lean" / data_root_workspace_key(data_root)
            write_files.assert_called_once_with(relocated)
            set_flags.assert_called_once()
            self.assertTrue(set_flags.call_args.kwargs["enabled"])
            self.assertEqual(set_flags.call_args.kwargs["workspace_dir"], str(relocated.resolve()))
            self.assertFalse(old_workspace.exists())

    def test_install_lean4_keeps_unrecognised_folder_when_relocating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / ("deep" * 15) / "backend" / "data"
            unrelated = data_root / "lean4_workspace"
            unrelated.mkdir(parents=True)
            (unrelated / "notes.txt").write_text("not a MOTO workspace", encoding="utf-8")
            runtime = mock.Mock(data_root=str(data_root))

            with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(root / "appdata")}):
                self._install_lean4_on_windows(runtime, {})

            self.assertTrue((unrelated / "notes.txt").exists())

    def test_install_lean4_disables_proofs_when_windows_workspace_path_is_too_long(self) -> None:
        deep_install = r"C:\Users\someone\Downloads\MOTO-Autonomous-ASI-dev\MOTO-Autonomous-ASI-dev"
        runtime = mock.Mock(data_root=deep_install + r"\backend\data")
        env: dict[str, str] = {}
        # No per-user folder to relocate into, so the path check is the fallback.
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": ""}), mock.patch.object(
            moto_launcher.sys, "platform", "win32"
        ):
            with mock.patch.object(moto_launcher, "ensure_git_for_lean", return_value="git"):
                with mock.patch.object(moto_launcher, "get_lean_command", return_value=r"C:\elan\lean.exe"):
                    with mock.patch.object(moto_launcher, "get_lake_command", return_value=r"C:\elan\lake.exe"):
                        with mock.patch.object(
                            moto_launcher.subprocess,
                            "check_output",
                            return_value="Lean (version 4.24.0)",
                        ):
                            with mock.patch.object(moto_launcher.subprocess, "run"):
                                with mock.patch.object(
                                    moto_launcher, "_write_lean_workspace_files"
                                ) as write_files:
                                    with mock.patch.object(moto_launcher, "_set_lean_env_flags") as set_flags:
                                        moto_launcher.install_lean4(runtime, env)

        set_flags.assert_called_once_with(
            env,
            enabled=False,
            unavailable_reason="lean_workspace_path_too_long",
        )
        write_files.assert_not_called()

    def test_check_node_installation_prepends_detected_node_dir_for_npm_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            node_dir = Path(temp_dir) / "nodejs"
            node_dir.mkdir()
            node_path = str(node_dir / "node.exe")
            npm_path = str(node_dir / "npm.cmd")
            Path(node_path).write_text("", encoding="utf-8")
            Path(npm_path).write_text("", encoding="utf-8")

            with mock.patch.dict(os.environ, {"PATH": r"C:\Windows\System32"}, clear=False):
                with mock.patch.object(moto_launcher.sys, "platform", "win32"):
                    with mock.patch.object(moto_launcher, "get_node_command", return_value=node_path):
                        with mock.patch.object(moto_launcher, "get_npm_command", return_value=npm_path):
                            with mock.patch.object(moto_launcher.subprocess, "check_output", side_effect=["v24.16.0", "11.13.0"]):
                                moto_launcher.check_node_installation()

                self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(node_dir.resolve()))

    def test_chromadb_native_probe_uses_clean_child_interpreter(self) -> None:
        result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(moto_launcher, "get_python_command", return_value="python"):
            with mock.patch.object(moto_launcher.subprocess, "run", return_value=result) as run:
                self.assertEqual(moto_launcher._probe_chromadb_native_bindings(), (True, ""))

        run.assert_called_once_with(
            ["python", "-c", "import chromadb_rust_bindings"],
            cwd=str(moto_launcher.SCRIPT_DIR),
            stdout=moto_launcher.subprocess.PIPE,
            stderr=moto_launcher.subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def test_ensure_native_dependencies_installs_vc_runtime_and_reprobes(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_probe_chromadb_native_bindings",
                side_effect=[(False, "missing DLL"), (True, "")],
            ) as probe:
                with mock.patch.object(
                    moto_launcher,
                    "install_windows_vc_runtime",
                    return_value=True,
                ) as installer:
                    moto_launcher.ensure_python_native_dependencies()

        installer.assert_called_once_with()
        self.assertEqual(probe.call_count, 2)

    def test_install_windows_vc_runtime_uses_winget_machine_package(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(moto_launcher, "resolve_command", return_value="winget"):
                with mock.patch.object(moto_launcher, "_windows_python_architecture", return_value="x64"):
                    with mock.patch.object(moto_launcher, "run_visible", return_value=0) as run_visible:
                        self.assertTrue(moto_launcher.install_windows_vc_runtime())

        self.assertEqual(
            run_visible.call_args_list[1].args[0],
            [
                "winget",
                "install",
                "--id",
                "Microsoft.VCRedist.2015+.x64",
                "-e",
                "--source",
                "winget",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
            ],
        )

    def test_ensure_native_dependencies_fails_before_backend_when_repair_fails(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_probe_chromadb_native_bindings",
                side_effect=[
                    (False, "DLL load failed: initial dependency"),
                    (False, "DLL load failed: missing dependency"),
                ],
            ):
                with mock.patch.object(
                    moto_launcher,
                    "install_windows_vc_runtime",
                    return_value=False,
                ):
                    with mock.patch.object(moto_launcher, "_windows_python_architecture", return_value="x64"):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "vc_redist.x64.exe.*missing dependency",
                        ):
                            moto_launcher.ensure_python_native_dependencies()

    def test_ensure_native_dependencies_reprobes_when_winget_reports_failure(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_probe_chromadb_native_bindings",
                side_effect=[(False, "initial failure"), (True, "")],
            ) as probe:
                with mock.patch.object(
                    moto_launcher,
                    "install_windows_vc_runtime",
                    return_value=False,
                ):
                    moto_launcher.ensure_python_native_dependencies()

        self.assertEqual(probe.call_count, 2)

    def test_windows_python_architecture_prefers_interpreter_platform(self) -> None:
        with mock.patch.object(moto_launcher.sysconfig, "get_platform", return_value="win-arm64"):
            with mock.patch.object(moto_launcher.platform, "machine", return_value="AMD64"):
                self.assertEqual(moto_launcher._windows_python_architecture(), "arm64")

    def test_ensure_native_dependencies_skips_installer_when_probe_succeeds(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "win32"):
            with mock.patch.object(
                moto_launcher,
                "_probe_chromadb_native_bindings",
                return_value=(True, ""),
            ):
                with mock.patch.object(moto_launcher, "install_windows_vc_runtime") as installer:
                    moto_launcher.ensure_python_native_dependencies()

        installer.assert_not_called()

    def test_ensure_native_dependencies_is_noop_outside_windows(self) -> None:
        with mock.patch.object(moto_launcher.sys, "platform", "linux"):
            with mock.patch.object(moto_launcher, "_probe_chromadb_native_bindings") as probe:
                moto_launcher.ensure_python_native_dependencies()

        probe.assert_not_called()

    def test_frontend_dependency_install_uses_lockfile_and_read_only_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            frontend_dir = repo_root / "frontend"
            frontend_dir.mkdir()
            (frontend_dir / "package-lock.json").write_text("{}", encoding="utf-8")

            install_result = mock.Mock(returncode=0, stdout="added 1 package")
            audit_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, audit_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "ci"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "--audit-level=high"])
            self.assertTrue((frontend_dir / "node_modules" / ".moto_package_lock.sha256").exists())

    def test_frontend_dependency_install_reconciles_existing_modules_non_destructively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            frontend_dir = repo_root / "frontend"
            (frontend_dir / "node_modules").mkdir(parents=True)
            (frontend_dir / "package-lock.json").write_text("{}", encoding="utf-8")

            install_result = mock.Mock(returncode=0, stdout="up to date")
            audit_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, audit_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "install", "--no-save"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "--audit-level=high"])

    def test_frontend_dependency_install_skips_reinstall_when_lock_marker_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            frontend_dir = repo_root / "frontend"
            node_modules_dir = frontend_dir / "node_modules"
            node_modules_dir.mkdir(parents=True)
            package_lock = frontend_dir / "package-lock.json"
            package_lock.write_text("{}", encoding="utf-8")
            (node_modules_dir / ".moto_package_lock.sha256").write_text(
                moto_launcher.file_sha256(package_lock),
                encoding="utf-8",
            )

            audit_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", return_value=audit_result) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["npm", "audit", "--audit-level=high"])

    def test_frontend_dependency_install_warns_on_high_audit_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            frontend_dir = repo_root / "frontend"
            frontend_dir.mkdir()
            (frontend_dir / "package-lock.json").write_text("{}", encoding="utf-8")

            install_result = mock.Mock(returncode=0, stdout="added 1 package")
            audit_result = mock.Mock(returncode=1, stdout="1 high severity vulnerability")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, audit_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertTrue(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "ci"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "--audit-level=high"])

    def test_frontend_dependency_install_reports_windows_file_lock_with_existing_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            frontend_dir = repo_root / "frontend"
            (frontend_dir / "node_modules").mkdir(parents=True)
            (frontend_dir / "package-lock.json").write_text("{}", encoding="utf-8")

            install_result = mock.Mock(
                returncode=-4048,
                stdout="npm error code EPERM\nnpm error syscall unlink\noperation not permitted",
            )
            audit_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, audit_result]) as run:
                        with mock.patch.object(moto_launcher, "exit_with_pause", side_effect=RuntimeError("exit")):
                            with self.assertRaisesRegex(RuntimeError, "exit"):
                                moto_launcher.install_frontend_dependencies()

            run.assert_called_once()
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "install", "--no-save"])

    def test_frontend_dependency_install_falls_back_when_lockfile_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "frontend").mkdir()

            install_result = mock.Mock(returncode=0, stdout="added 1 package")
            audit_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, audit_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "install"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "--audit-level=high"])


class LinuxLauncherStrategyTests(TestCase):
    def test_using_repo_local_venv_detects_repo_scoped_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            python_path = repo_root / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_python_command", return_value=str(python_path)):
                    self.assertTrue(moto_launcher.using_repo_local_venv())

    def test_launch_service_uses_linux_terminal_when_available(self) -> None:
        process = mock.Mock(pid=3210)
        with mock.patch.object(moto_launcher.sys, "platform", "linux"):
            with mock.patch.object(moto_launcher, "resolve_linux_terminal", return_value=("gnome-terminal", "/usr/bin/gnome-terminal")):
                with mock.patch.object(moto_launcher.subprocess, "Popen", return_value=process) as popen:
                    service = moto_launcher.launch_service(
                        title="MOTO Backend [default]",
                        service_slug="backend",
                        args=["python3", "-m", "uvicorn"],
                        cwd="/tmp/project",
                        env={},
                        log_root="/tmp/project/logs",
                    )

        self.assertEqual(service.mode, "terminal")
        self.assertEqual(service.pid, 3210)
        self.assertIsNone(service.log_path)
        popen.assert_called_once()

    def test_launch_service_falls_back_to_background_when_no_linux_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            process = mock.Mock(pid=4242)
            with mock.patch.object(moto_launcher.sys, "platform", "linux"):
                with mock.patch.object(moto_launcher, "resolve_linux_terminal", return_value=None):
                    with mock.patch.object(moto_launcher.subprocess, "Popen", return_value=process) as popen:
                        service = moto_launcher.launch_service(
                            title="MOTO Backend [default]",
                            service_slug="backend",
                            args=["python3", "-m", "http.server"],
                            cwd=temp_dir,
                            env={},
                            log_root=temp_dir,
                        )

        self.assertEqual(service.mode, "background")
        self.assertEqual(service.pid, 4242)
        self.assertIsNotNone(service.log_path)
        self.assertTrue(service.log_path.endswith("launcher_backend.log"))
        popen.assert_called_once()


class ArchiveExtractionTests(TestCase):
    def test_extract_archive_rejects_tar_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "archive.tar.gz"
            destination = root / "extract"
            outside = root / "evil.txt"

            with tarfile.open(archive_path, "w:gz") as archive:
                data = b"bad"
                member = tarfile.TarInfo("../evil.txt")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

            with self.assertRaises(RuntimeError):
                moto_launcher._extract_archive(archive_path, destination)

            self.assertFalse(outside.exists())

    def test_extract_archive_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "archive.zip"
            destination = root / "extract"
            outside = root / "evil.txt"

            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../evil.txt", "bad")

            with self.assertRaises(RuntimeError):
                moto_launcher._extract_archive(archive_path, destination)

            self.assertFalse(outside.exists())


if __name__ == "__main__":
    main()
