import json
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, main, mock

from backend.api.routes import features as features_route
from backend.api.routes import update as update_route
from moto_updater import BuildManifest, InstallState, UpdateCheckResult


class UpdateRouteGitPullTests(IsolatedAsyncioTestCase):
    def _manifest(self, version: str, commit: str) -> BuildManifest:
        return BuildManifest(
            version=version,
            build_commit=commit,
            update_channel="main",
            api_contract_version="build5-v29",
        )

    async def test_git_pull_route_refuses_dirty_tracked_checkout(self) -> None:
        local_manifest = self._manifest("1.0.9", "local")
        remote_manifest = self._manifest("1.0.9", "remote")
        run_git_command = mock.AsyncMock(return_value=(0, " M moto_updater.py"))

        with mock.patch("moto_updater.load_local_manifest", return_value=local_manifest):
            with mock.patch("moto_updater.fetch_remote_manifest", return_value=remote_manifest):
                with mock.patch.object(update_route, "_run_git_command", run_git_command):
                    await update_route._run_git_pull()

        self.assertEqual(update_route._pull_state["status"], "error")
        self.assertEqual(update_route._pull_state["returncode"], 1)
        self.assertIn("local modifications", "\n".join(update_route._pull_state["output_lines"]))
        run_git_command.assert_awaited_once_with("status", "--porcelain", "--untracked-files=no")

    async def test_git_pull_route_uses_fetch_and_ff_only_merge(self) -> None:
        local_manifest = self._manifest("1.0.9", "local")
        remote_manifest = self._manifest("1.0.9", "remote")
        run_git_command = mock.AsyncMock(
            side_effect=[
                (0, ""),
                (0, ""),
                (0, "0\t1"),
                (0, "Updating local..remote\nFast-forward"),
            ]
        )

        with mock.patch("moto_updater.load_local_manifest", return_value=local_manifest):
            with mock.patch("moto_updater.fetch_remote_manifest", return_value=remote_manifest):
                with mock.patch.object(update_route, "_run_git_command", run_git_command):
                    await update_route._run_git_pull()

        self.assertEqual(update_route._pull_state["status"], "done")
        self.assertEqual(update_route._pull_state["returncode"], 0)
        self.assertEqual(
            [call.args for call in run_git_command.await_args_list],
            [
                ("status", "--porcelain", "--untracked-files=no"),
                ("fetch", "origin", "main", "--quiet"),
                ("rev-list", "--left-right", "--count", "HEAD...origin/main"),
                ("merge", "--ff-only", "origin/main"),
            ],
        )


class RuntimeUpdateNoticeTests(IsolatedAsyncioTestCase):
    def _update_result(self, local_version: str = "1.0.9", remote_version: str = "1.0.9") -> UpdateCheckResult:
        return UpdateCheckResult(
            BuildManifest(
                version=local_version,
                build_commit="localcommit",
                update_channel="main",
                api_contract_version="build5-v29",
            ),
            BuildManifest(
                version=remote_version,
                build_commit="remotecommit",
                update_channel="main",
                api_contract_version="build5-v29",
            ),
            InstallState(
                kind="clean_git_clone",
                label="Clean git clone on main",
                can_auto_apply=True,
                reason="Clean git checkout tracking origin/main.",
            ),
            metadata_source="manifest",
        )

    async def test_update_notice_returns_existing_notice_without_runtime_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notice_path = Path(temp_dir) / ".moto_update_notice.json"
            notice_path.write_text(
                json.dumps({"update_available": True, "available_version": "1.0.10"}),
                encoding="utf-8",
            )
            refresh = mock.AsyncMock()

            with mock.patch.object(features_route, "_UPDATE_NOTICE_PATH", notice_path):
                with mock.patch.object(features_route, "_refresh_runtime_update_notice_if_due", refresh):
                    result = await features_route.get_update_notice()

        self.assertTrue(result["update_available"])
        refresh.assert_not_awaited()

    async def test_update_notice_refreshes_due_runtime_check_and_returns_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notice_path = Path(temp_dir) / ".moto_update_notice.json"
            update_result = self._update_result()

            def write_notice(result: UpdateCheckResult) -> None:
                self.assertIs(result, update_result)
                notice_path.write_text(
                    json.dumps({"update_available": True, "available_version": "1.0.9"}),
                    encoding="utf-8",
                )

            with mock.patch.object(features_route, "_UPDATE_NOTICE_PATH", notice_path):
                with mock.patch.object(features_route, "_UPDATE_NOTICE_REFRESH_INTERVAL_SECONDS", 0):
                    with mock.patch.object(features_route._update_notice_refresh_state, "last_refresh_at", 0):
                        with mock.patch.object(features_route.system_config, "generic_mode", False):
                            with mock.patch.object(features_route.system_config, "instance_id", "current"):
                                with mock.patch("moto_updater.check_for_updates", return_value=update_result) as check:
                                    with mock.patch("moto_updater.write_update_notice", side_effect=write_notice) as write:
                                        result = await features_route.get_update_notice()

        self.assertTrue(result["update_available"])
        check.assert_called_once_with(exclude_instance_id="current")
        write.assert_called_once_with(update_result)

    async def test_update_notice_refresh_respects_local_version_newer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notice_path = Path(temp_dir) / ".moto_update_notice.json"
            update_result = self._update_result(local_version="1.0.10", remote_version="1.0.9")

            def write_notice(result: UpdateCheckResult) -> None:
                self.assertFalse(result.update_available)

            with mock.patch.object(features_route, "_UPDATE_NOTICE_PATH", notice_path):
                with mock.patch.object(features_route, "_UPDATE_NOTICE_REFRESH_INTERVAL_SECONDS", 0):
                    with mock.patch.object(features_route._update_notice_refresh_state, "last_refresh_at", 0):
                        with mock.patch.object(features_route.system_config, "generic_mode", False):
                            with mock.patch("moto_updater.check_for_updates", return_value=update_result):
                                with mock.patch("moto_updater.write_update_notice", side_effect=write_notice):
                                    result = await features_route.get_update_notice()

        self.assertFalse(result["update_available"])


class UpdateVersionComparisonTests(IsolatedAsyncioTestCase):
    async def test_downgrade_guard_treats_zero_padded_release_as_same_version(self) -> None:
        self.assertFalse(update_route._is_downgrade("1.1.00", "1.1.0"))
        self.assertFalse(update_route._is_downgrade("1.1.0", "1.1.00"))
        self.assertTrue(update_route._is_downgrade("1.1.00", "1.0.99"))


if __name__ == "__main__":
    main()
