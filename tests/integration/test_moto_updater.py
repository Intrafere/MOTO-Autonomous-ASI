import base64
import json
from pathlib import Path
import tempfile
import urllib.error
from unittest import TestCase, main, mock
import zipfile

import moto_updater


class RepoSlugTests(TestCase):
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

    def test_normalize_repo_slug_rejects_lookalike_or_untrusted_hosts(self) -> None:
        cases = [
            "https://github.com.evil/Intrafere/MOTO-Autonomous-ASI",
            "https://evil.example/github.com/Intrafere/MOTO-Autonomous-ASI",
            "ssh://git@github.com.evil/Intrafere/MOTO-Autonomous-ASI.git",
            "https://example.com/Intrafere/MOTO-Autonomous-ASI",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertIsNone(moto_updater._normalize_repo_slug(raw))


class InstallStateTests(TestCase):
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
            version="1.0.9",
            build_commit="localcommit",
            update_channel="main",
            api_contract_version="build5-v1",
        )
        fallback_manifest = moto_updater.BuildManifest(
            version="1.0.9",
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

    def test_check_for_updates_formats_transient_github_failure(self) -> None:
        local_manifest = moto_updater.BuildManifest(
            version="1.0.9",
            build_commit="localcommit",
            update_channel="main",
            api_contract_version="build5-v1",
        )
        http_504 = urllib.error.HTTPError(
            url="https://api.github.com/repos/example/repo/branches/main",
            code=504,
            msg="Gateway Time-out",
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
                    with mock.patch.object(moto_updater, "fetch_remote_manifest", side_effect=http_504):
                        with mock.patch.object(moto_updater, "fetch_branch_head_fallback") as fallback:
                            result = moto_updater.check_for_updates()

        self.assertFalse(result.update_available)
        self.assertEqual(result.metadata_source, "none")
        self.assertIsNotNone(result.error)
        self.assertIn("GitHub update metadata is temporarily unavailable", result.error or "")
        self.assertIn("Startup will continue", result.error or "")
        fallback.assert_not_called()

    def test_fetch_remote_manifest_uses_branch_head_as_update_key(self) -> None:
        local_manifest = moto_updater.BuildManifest(
            version="1.0.9",
            build_commit="localcommit",
            update_channel="main",
            api_contract_version="build5-v1",
        )
        manifest_payload = {
            "manifest_version": 1,
            "version": "1.0.9",
            "build_commit": "stale-manifest-commit",
            "update_channel": "main",
            "api_contract_version": "build5-v29",
        }
        branch_payload = {"commit": {"sha": "actual-branch-head"}}

        with mock.patch.object(moto_updater, "_fetch_repo_file_json", return_value=manifest_payload) as fetch_file:
            with mock.patch.object(moto_updater, "_fetch_json_url", return_value=branch_payload):
                remote_manifest = moto_updater.fetch_remote_manifest(local_manifest)

        self.assertEqual(remote_manifest.version, "1.0.9")
        self.assertEqual(remote_manifest.build_commit, "actual-branch-head")
        self.assertEqual(remote_manifest.api_contract_version, "build5-v29")
        fetch_file.assert_called_once_with(
            "actual-branch-head",
            "moto-update-manifest.json",
            10,
        )

    def test_fetch_repo_file_json_uses_contents_api_payload(self) -> None:
        file_payload = {
            "type": "file",
            "encoding": "base64",
            "content": base64.b64encode(b'{"version": "1.0.9"}').decode("ascii"),
        }

        with mock.patch.object(moto_updater, "_fetch_json_url", return_value=file_payload) as fetch_json:
            with mock.patch.object(
                moto_updater,
                "_contents_api_url_for_path",
                return_value="https://api.github.com/repos/owner/repo/contents/package.json?ref=main",
            ):
                payload = moto_updater._fetch_repo_file_json("main", "package.json", 10)

        self.assertEqual(payload["version"], "1.0.9")
        fetch_json.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/package.json?ref=main",
            10,
        )

    def test_load_local_manifest_uses_git_head_for_git_checkouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            manifest_path = repo_root / "moto-update-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.0.9",
                        "build_commit": "stale-local-commit",
                        "update_channel": "main",
                        "api_contract_version": "build5-v29",
                    }
                ),
                encoding="utf-8",
            )
            (repo_root / ".git").mkdir()

            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                with mock.patch.object(moto_updater, "LOCAL_MANIFEST_PATH", manifest_path):
                    with mock.patch.object(moto_updater, "_git_checkout_matches_repo", return_value=True):
                        with mock.patch.object(moto_updater, "_git_output", return_value=(0, "actual-local-head", "")):
                            local_manifest = moto_updater.load_local_manifest()

        self.assertEqual(local_manifest.build_commit, "actual-local-head")

    def test_load_local_manifest_uses_archival_head_for_zip_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            manifest_path = repo_root / "moto-update-manifest.json"
            archival_path = repo_root / ".git_archival.txt"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.0.9",
                        "build_commit": "stale-local-commit",
                        "update_channel": "main",
                        "api_contract_version": "build5-v29",
                    }
                ),
                encoding="utf-8",
            )
            archival_path.write_text(
                "node: 0123456789abcdef0123456789abcdef01234567\n",
                encoding="utf-8",
            )

            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                with mock.patch.object(moto_updater, "LOCAL_MANIFEST_PATH", manifest_path):
                    with mock.patch.object(moto_updater, "ARCHIVAL_METADATA_PATH", archival_path):
                        local_manifest = moto_updater.load_local_manifest()

        self.assertEqual(local_manifest.build_commit, "0123456789abcdef0123456789abcdef01234567")

    def test_load_local_manifest_ignores_unexpanded_archival_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            manifest_path = repo_root / "moto-update-manifest.json"
            archival_path = repo_root / ".git_archival.txt"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.0.9",
                        "build_commit": "manifest-commit",
                        "update_channel": "main",
                        "api_contract_version": "build5-v29",
                    }
                ),
                encoding="utf-8",
            )
            archival_path.write_text("node: $Format:%H$\n", encoding="utf-8")

            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                with mock.patch.object(moto_updater, "LOCAL_MANIFEST_PATH", manifest_path):
                    with mock.patch.object(moto_updater, "ARCHIVAL_METADATA_PATH", archival_path):
                        local_manifest = moto_updater.load_local_manifest()

        self.assertEqual(local_manifest.build_commit, "manifest-commit")


class UpdateAvailabilityTests(TestCase):
    def _result(self, local_version: str, remote_version: str) -> moto_updater.UpdateCheckResult:
        install_state = moto_updater.InstallState(
            kind="zip_install",
            label="ZIP / extracted consumer install",
            can_auto_apply=True,
            reason="ZIP / extracted consumer install.",
        )
        return moto_updater.UpdateCheckResult(
            moto_updater.BuildManifest(
                version=local_version,
                build_commit="localcommit",
                update_channel="main",
                api_contract_version="build5-v1",
            ),
            moto_updater.BuildManifest(
                version=remote_version,
                build_commit="remotecommit",
                update_channel="main",
                api_contract_version="build5-v1",
            ),
            install_state,
            metadata_source="manifest",
        )

    def test_update_available_false_when_local_version_is_newer(self) -> None:
        result = self._result("1.0.9", "1.0.8")

        self.assertFalse(result.update_available)
        self.assertFalse(result.can_apply_update)

    def test_update_available_uses_numeric_version_comparison(self) -> None:
        result = self._result("1.0.10", "1.0.9")

        self.assertFalse(result.update_available)

    def test_update_available_treats_zero_padded_release_as_same_version(self) -> None:
        result = self._result("1.1.00", "1.1.0")

        self.assertTrue(result.update_available)
        self.assertTrue(result.can_apply_update)
        self.assertEqual(moto_updater._compare_version_strings("1.1.00", "1.1.0"), 0)

    def test_update_available_still_true_for_same_version_new_commit(self) -> None:
        result = self._result("1.0.9", "1.0.9")

        self.assertTrue(result.update_available)
        self.assertTrue(result.can_apply_update)


class LauncherStateTests(TestCase):
    def test_nonsecret_json_metadata_rejects_credential_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metadata.json"

            for payload in (
                {"api_key": "example"},
                {"nested": {"refresh_token": "example"}},
                {"instances": [{"authorization": "Bearer example"}]},
            ):
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        moto_updater._write_nonsecret_json_metadata(output_path, payload)

            self.assertFalse(output_path.exists())

    def test_nonsecret_json_metadata_allows_keyring_namespace_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metadata.json"
            payload = {
                "instance_id": "instance_one",
                "keyring_namespace": "instance_one",
            }

            moto_updater._write_nonsecret_json_metadata(output_path, payload)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), payload)

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

    def test_cleanup_launcher_state_can_exclude_current_instance_from_returned_active_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / ".moto_launcher_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "instances": [
                            {"instance_id": "current", "backend_window_pid": 100, "frontend_window_pid": 101},
                            {"instance_id": "other", "backend_window_pid": 200, "frontend_window_pid": 201},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_is_pid_running(pid: int | None) -> bool:
                return pid in {100, 101, 200, 201}

            with mock.patch.object(moto_updater, "LAUNCHER_STATE_PATH", state_path):
                with mock.patch.object(moto_updater, "_is_pid_running", side_effect=fake_is_pid_running):
                    instances = moto_updater.cleanup_launcher_state(exclude_instance_id="current")
                    saved_payload = json.loads(state_path.read_text(encoding="utf-8"))

            self.assertEqual([instance["instance_id"] for instance in instances], ["other"])
            self.assertEqual(
                [instance["instance_id"] for instance in saved_payload["instances"]],
                ["current", "other"],
            )

    def test_launcher_state_writer_persists_only_public_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / ".moto_launcher_state.json"
            payload = {
                "instances": [{
                    "instance_id": "instance_one",
                    "backend_window_pid": 100,
                    "frontend_window_pid": 101,
                    "backend_port": 8000,
                    "frontend_port": 5173,
                    "data_root": "data",
                    "log_root": "logs",
                    "storage_prefix": "instance_one",
                    "unknown_field": "discard-me",
                    "keyring_namespace": "discard-selector",
                }],
                "unknown_top_level": "discard-me",
            }

            with mock.patch.object(moto_updater, "LAUNCHER_STATE_PATH", state_path):
                moto_updater._save_launcher_state(payload)
                saved = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(set(saved), {"instances"})
        self.assertEqual(
            set(saved["instances"][0]),
            {
                "instance_id",
                "backend_window_pid",
                "frontend_window_pid",
                "backend_port",
                "frontend_port",
                "data_root",
                "log_root",
                "storage_prefix",
            },
        )

    def test_last_instance_record_does_not_persist_keyring_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir) / ".moto_last_instance.json"

            with mock.patch.object(moto_updater, "LAUNCHER_LAST_INSTANCE_PATH", record_path):
                moto_updater.save_last_instance_record(
                    instance_id="instance_one",
                    data_root="data",
                    log_root="logs",
                    storage_prefix="instance_one",
                )
                payload = json.loads(record_path.read_text(encoding="utf-8"))

        self.assertNotIn("keyring_namespace", payload)
        self.assertNotIn("secret_namespace", payload)

    def test_last_instance_record_reads_legacy_secret_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir) / ".moto_last_instance.json"
            record_path.write_text(
                json.dumps(
                    {
                        "instance_id": "instance_one",
                        "data_root": "data",
                        "log_root": "logs",
                        "secret_namespace": "legacy_namespace",
                        "storage_prefix": "instance_one",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(moto_updater, "LAUNCHER_LAST_INSTANCE_PATH", record_path):
                payload = moto_updater.load_last_instance_record()

        self.assertIsNotNone(payload)
        self.assertEqual(payload["keyring_namespace"], "legacy_namespace")
        self.assertNotIn("secret_namespace", payload)

    def test_updater_extract_archive_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "archive.zip"
            destination = root / "extract"
            outside = root / "evil.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../evil.txt", "bad")

            with self.assertRaises(RuntimeError):
                moto_updater._extract_archive(archive_path, destination)

            self.assertFalse(outside.exists())


class SnapshotSyncTests(TestCase):
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

    def test_apply_zip_update_writes_resolved_manifest_after_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "install"
            repo_root.mkdir()
            manifest_path = repo_root / "moto-update-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.0.9",
                        "build_commit": "old-local",
                        "update_channel": "main",
                        "api_contract_version": "build5-v1",
                    }
                ),
                encoding="utf-8",
            )

            def fake_download_archive(remote_manifest: moto_updater.BuildManifest, archive_path: Path) -> None:
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(
                        "MOTO/moto-update-manifest.json",
                        json.dumps(
                            {
                                "manifest_version": 1,
                                "version": remote_manifest.version,
                                "build_commit": "stale-inner-manifest",
                                "update_channel": remote_manifest.update_channel,
                                "api_contract_version": remote_manifest.api_contract_version,
                            }
                        ),
                    )
                    archive.writestr("MOTO/moto_launcher.py", "new launcher\n")

            remote_manifest = moto_updater.BuildManifest(
                version="1.0.9",
                build_commit="resolved-remote-head",
                update_channel="main",
                api_contract_version="build5-v29",
            )

            with mock.patch.object(moto_updater, "REPO_ROOT", repo_root):
                with mock.patch.object(moto_updater, "LOCAL_MANIFEST_PATH", manifest_path):
                    with mock.patch.object(moto_updater, "cleanup_launcher_state", return_value=[]):
                        with mock.patch.object(moto_updater, "_download_archive", side_effect=fake_download_archive):
                            with mock.patch.object(moto_updater, "_relaunch_launcher", return_value=None):
                                applied, message = moto_updater.apply_zip_update(
                                    remote_manifest=remote_manifest,
                                    launcher_args=[],
                                    env={},
                                )

            installed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(applied, message)
        self.assertEqual(installed_manifest["version"], "1.0.9")
        self.assertEqual(installed_manifest["build_commit"], "resolved-remote-head")


class RelaunchCommandTests(TestCase):
    def test_build_relaunch_command_prefers_linux_entrypoint_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            entrypoint = Path(temp_dir) / "linux-ubuntu-launcher.sh"
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
    main()
