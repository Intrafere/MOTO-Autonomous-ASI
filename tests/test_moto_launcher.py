import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import moto_launcher


class ResolveInstanceRuntimeTests(unittest.TestCase):
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

    def test_occupied_defaults_allocate_isolated_instance(self) -> None:
        def fake_port_in_use(port: int) -> bool:
            return port in {8000, 5173}

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=None):
                with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                    with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_test_1234"):
                        runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "instance_test_1234")
        self.assertEqual(runtime.backend_port, 8001)
        self.assertEqual(runtime.frontend_port, 5174)
        self.assertFalse(runtime.is_default)
        self.assertIn(".moto_instances", runtime.data_root)
        self.assertIn("instance_test_1234", runtime.data_root)
        self.assertIn(".moto_instances", runtime.log_root)
        self.assertEqual(runtime.secret_namespace, "instance_test_1234")
        self.assertEqual(runtime.storage_prefix, "instance_test_1234")
        self.assertFalse(runtime.explicit_override)

    def test_last_record_default_is_reused_when_ports_busy(self) -> None:
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
            return port in {8000, 5173}

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
        self.assertNotEqual(runtime.frontend_port, 5173)

    def test_last_record_isolated_instance_is_reused_even_when_default_ports_are_free(self) -> None:
        """A prior isolated launch must keep its namespace even when default ports become free."""
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

        self.assertEqual(runtime.instance_id, "instance_20260101_000000_1111")
        self.assertFalse(runtime.is_default)
        self.assertEqual(runtime.secret_namespace, "instance_20260101_000000_1111")
        self.assertEqual(runtime.storage_prefix, "instance_20260101_000000_1111")

    def test_live_instance_is_not_reused_to_avoid_data_root_collision(self) -> None:
        """A recorded identity currently live in another process must be avoided."""
        saved_record = {
            "instance_id": "instance_20260101_000000_1111",
            "data_root": None,
            "log_root": None,
            "secret_namespace": "instance_20260101_000000_1111",
            "storage_prefix": "instance_20260101_000000_1111",
        }
        live_record = [{"instance_id": "instance_20260101_000000_1111"}]

        def fake_port_in_use(port: int) -> bool:
            return port in {8000, 5173}

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=live_record):
                    with mock.patch.object(moto_launcher, "port_in_use", side_effect=fake_port_in_use):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_test_freshly_minted"):
                            runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "instance_test_freshly_minted")
        self.assertNotEqual(runtime.instance_id, saved_record["instance_id"])
        self.assertFalse(runtime.is_default)

    def test_live_default_instance_is_not_recreated_when_ports_look_free(self) -> None:
        """A live recorded default instance must block fallback to the default identity."""
        saved_record = {
            "instance_id": "default",
            "data_root": None,
            "log_root": None,
            "secret_namespace": None,
            "storage_prefix": None,
        }
        live_record = [{"instance_id": "default"}]

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=live_record):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_safe_parallel"):
                            runtime = moto_launcher.resolve_instance_runtime()

        self.assertEqual(runtime.instance_id, "instance_safe_parallel")
        self.assertFalse(runtime.is_default)
        self.assertEqual(runtime.secret_namespace, "instance_safe_parallel")
        self.assertEqual(runtime.storage_prefix, "instance_safe_parallel")

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


class WindowsLauncherStrategyTests(unittest.TestCase):
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


class LinuxLauncherStrategyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
