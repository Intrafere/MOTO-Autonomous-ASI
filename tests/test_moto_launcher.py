import io
import os
from pathlib import Path
import tarfile
import tempfile
from unittest import TestCase, main, mock
import zipfile

import moto_launcher


class ResolveInstanceRuntimeTests(TestCase):
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
        live_record = [{"instance_id": "default"}]

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(moto_launcher, "load_last_instance_record", return_value=saved_record):
                with mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=live_record):
                    with mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                        with mock.patch.object(moto_launcher, "new_instance_id", return_value="instance_safe_parallel"):
                            with self.assertRaisesRegex(RuntimeError, "default MOTO instance already appears to be running"):
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

    def test_runtime_lock_blocks_live_default_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 4242, "default")
            with mock.patch.object(moto_launcher, "is_pid_running", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "data root is already in use"):
                    moto_launcher.assert_runtime_lock_available(temp_dir)

    def test_runtime_lock_ignores_stale_default_backend_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            moto_launcher.write_runtime_lock(temp_dir, 4242, "default")
            with mock.patch.object(moto_launcher, "is_pid_running", return_value=False):
                moto_launcher.assert_runtime_lock_available(temp_dir)


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

    def test_frontend_dependency_install_runs_audit_fix_for_clean_git_vulnerabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "frontend").mkdir()

            install_result = mock.Mock(
                returncode=0,
                stdout="added 1 package, and audited 1 package\n1 high severity vulnerability",
            )
            fix_result = mock.Mock(returncode=0, stdout="fixed 1 vulnerability")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, fix_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "install"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "fix"])

    def test_frontend_dependency_install_runs_audit_fix_for_npm_remediation_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "frontend").mkdir()

            install_result = mock.Mock(
                returncode=0,
                stdout=(
                    "27 packages are looking for funding\n"
                    "  run `npm fund` for details\n\n"
                    "2 vulnerabilities (1 moderate, 1 high)\n\n"
                    "To address all issues, run:\n"
                    "  npm audit fix\n\n"
                    "Run `npm audit` for details.\n"
                ),
            )
            fix_result = mock.Mock(returncode=0, stdout="found 0 vulnerabilities")

            with mock.patch.object(moto_launcher, "SCRIPT_DIR", repo_root):
                with mock.patch.object(moto_launcher, "get_npm_command", return_value="npm"):
                    with mock.patch.object(moto_launcher.subprocess, "run", side_effect=[install_result, fix_result]) as run:
                        _, vulnerability_warning = moto_launcher.install_frontend_dependencies()

            self.assertFalse(vulnerability_warning)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["npm", "install"])
            self.assertEqual(run.call_args_list[1].args[0], ["npm", "audit", "fix"])


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
