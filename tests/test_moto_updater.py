import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest import mock

import moto_updater


class RepoSlugTests(unittest.TestCase):
    def test_normalize_repo_slug_handles_common_github_formats(self) -> None:
        cases = {
            "https://github.com/Intrafere/MOTO-Autonomous-ASI": "Intrafere/MOTO-Autonomous-ASI",
            "https://github.com/Intrafere/MOTO-Autonomous-ASI.git": "Intrafere/MOTO-Autonomous-ASI",
            "git@github.com:Intrafere/MOTO-Autonomous-ASI.git": "Intrafere/MOTO-Autonomous-ASI",
            "git+https://github.com/Intrafere/MOTO-Autonomous-ASI.git": "Intrafere/MOTO-Autonomous-ASI",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(moto_updater._normalize_repo_slug(raw), expected)


class InstallStateTests(unittest.TestCase):
    def test_classify_zip_install_when_repo_has_no_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                with mock.patch.object(moto_updater, "_git_checkout_matches_repo", return_value=False):
                    state = moto_updater.classify_install_state([])

        self.assertEqual(state.kind, "zip_install")
        self.assertTrue(state.can_auto_apply)

    def test_classify_clean_git_clone_when_checkout_is_safe(self) -> None:
        git_outputs = [
            (0, "main", ""),
            (0, "origin/main", ""),
            (0, "", ""),
            (0, "https://github.com/Intrafere/MOTO-Autonomous-ASI.git", ""),
        ]

        with mock.patch.object(moto_updater, "_git_checkout_matches_repo", return_value=True):
            with mock.patch.object(moto_updater, "_official_repo_slug", return_value="Intrafere/MOTO-Autonomous-ASI"):
                with mock.patch.object(moto_updater, "_git_output", side_effect=git_outputs):
                    state = moto_updater.classify_install_state([])

        self.assertEqual(state.kind, "clean_git_clone")
        self.assertTrue(state.can_auto_apply)
        self.assertEqual(state.git_branch, "main")
        self.assertEqual(state.git_upstream, "origin/main")

    def test_check_for_updates_falls_back_to_branch_head_when_manifest_missing(self) -> None:
        local_manifest = moto_updater.BuildManifest(
            version="1.0.7",
            build_commit="localcommit",
            update_channel="main",
            api_contract_version="build5-v1",
        )
        fallback_manifest = moto_updater.BuildManifest(
            version="1.0.6",
            build_commit="remotecommit",
            update_channel="main",
            api_contract_version="build5-v1",
        )
        http_404 = urllib.error.HTTPError(
            url="https://raw.githubusercontent.com/example/main/moto-update-manifest.json",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        with mock.patch.object(moto_updater, "load_local_manifest", return_value=local_manifest):
            with mock.patch.object(moto_updater, "cleanup_launcher_state", return_value=[]):
                with mock.patch.object(
                    moto_updater,
                    "classify_install_state",
                    return_value=moto_updater.InstallState(
                        kind="zip_install",
                        label="ZIP / extracted consumer install",
                        can_auto_apply=True,
                        reason="ZIP / extracted consumer install.",
                    ),
                ):
                    with mock.patch.object(moto_updater, "fetch_remote_manifest", side_effect=http_404):
                        with mock.patch.object(moto_updater, "fetch_branch_head_fallback", return_value=fallback_manifest):
                            result = moto_updater.check_for_updates()

        self.assertTrue(result.update_available)
        self.assertEqual(result.metadata_source, "branch_head_fallback")
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.warning)
        self.assertFalse(result.can_apply_update)


class LauncherStateTests(unittest.TestCase):
    def test_cleanup_launcher_state_removes_dead_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / ".moto_launcher_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "instances": [
                            {"instance_id": "alive", "backend_window_pid": 100, "frontend_window_pid": 101},
                            {"instance_id": "dead", "backend_window_pid": 200, "frontend_window_pid": 201},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_is_pid_running(pid: int | None) -> bool:
                return pid in {100, 101}

            with mock.patch.object(moto_updater, "LAUNCHER_STATE_PATH", state_path):
                with mock.patch.object(moto_updater, "_is_pid_running", side_effect=fake_is_pid_running):
                    instances = moto_updater.cleanup_launcher_state()

            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0]["instance_id"], "alive")


class SnapshotSyncTests(unittest.TestCase):
    def test_collect_preserved_relatives_includes_explicit_instance_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            instance_data = repo_root / ".moto_instances" / "instance_alpha" / "data"
            instance_logs = repo_root / ".moto_instances" / "instance_alpha" / "logs"
            instance_data.mkdir(parents=True)
            instance_logs.mkdir(parents=True)

            env = {
                "MOTO_INSTANCE_ID": "instance_alpha",
                "MOTO_DATA_ROOT": str(instance_data),
                "MOTO_LOG_ROOT": str(instance_logs),
            }
            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                preserved = moto_updater.collect_preserved_relatives(env, active_instances=[])

        self.assertIn(".moto_instances", preserved)
        self.assertIn(".moto_instances/instance_alpha", preserved)
        self.assertIn(".moto_instances/instance_alpha/data", preserved)
        self.assertIn(".moto_instances/instance_alpha/logs", preserved)

    def test_sync_snapshot_preserves_runtime_roots_and_can_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            destination_root = temp_root / "destination"
            backup_root = temp_root / "backup"

            (source_root / "backend" / "data").mkdir(parents=True)
            (source_root / "docs").mkdir(parents=True)
            (source_root / "moto_launcher.py").write_text("new launcher\n", encoding="utf-8")
            (source_root / "docs" / "guide.txt").write_text("new docs\n", encoding="utf-8")
            (source_root / "backend" / "data" / "keep.txt").write_text("new data\n", encoding="utf-8")

            (destination_root / "backend" / "data").mkdir(parents=True)
            (destination_root / "docs").mkdir(parents=True)
            (destination_root / "moto_launcher.py").write_text("old launcher\n", encoding="utf-8")
            (destination_root / "backend" / "data" / "keep.txt").write_text("original data\n", encoding="utf-8")

            journal = moto_updater.sync_snapshot_into_install(
                source_root=source_root,
                destination_root=destination_root,
                preserved_relatives={"backend/data"},
                backup_root=backup_root,
            )

            self.assertEqual((destination_root / "moto_launcher.py").read_text(encoding="utf-8"), "new launcher\n")
            self.assertEqual((destination_root / "docs" / "guide.txt").read_text(encoding="utf-8"), "new docs\n")
            self.assertEqual((destination_root / "backend" / "data" / "keep.txt").read_text(encoding="utf-8"), "original data\n")
            self.assertIn("moto_launcher.py", journal.overwritten_files)
            self.assertIn("docs/guide.txt", journal.created_files)

            moto_updater.restore_snapshot_from_backup(destination_root, backup_root, journal)

            self.assertEqual((destination_root / "moto_launcher.py").read_text(encoding="utf-8"), "old launcher\n")
            self.assertFalse((destination_root / "docs" / "guide.txt").exists())
            self.assertEqual((destination_root / "backend" / "data" / "keep.txt").read_text(encoding="utf-8"), "original data\n")


class RelaunchCommandTests(unittest.TestCase):
    def test_build_relaunch_command_prefers_linux_entrypoint_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            entrypoint = Path(temp_dir) / "Launch MOTO.sh"
            entrypoint.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            command = moto_updater._build_relaunch_command(
                {moto_updater.LAUNCHER_ENTRYPOINT_ENV: str(entrypoint)}
            )

        self.assertEqual(command, ["bash", str(entrypoint.resolve())])

    def test_build_relaunch_command_falls_back_to_python_launcher(self) -> None:
        command = moto_updater._build_relaunch_command({})

        self.assertEqual(command[0], moto_updater.sys.executable)
        self.assertTrue(command[1].endswith("moto_launcher.py"))


if __name__ == "__main__":
    unittest.main()
