import unittest
from unittest import mock

from backend.api.routes import update as update_route
from moto_updater import BuildManifest


class UpdateRouteGitPullTests(unittest.IsolatedAsyncioTestCase):
    def _manifest(self, version: str, commit: str) -> BuildManifest:
        return BuildManifest(
            version=version,
            build_commit=commit,
            update_channel="main",
            api_contract_version="build5-v12",
        )

    async def test_git_pull_route_refuses_dirty_tracked_checkout(self) -> None:
        local_manifest = self._manifest("1.0.7", "local")
        remote_manifest = self._manifest("1.0.8", "remote")
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
        local_manifest = self._manifest("1.0.7", "local")
        remote_manifest = self._manifest("1.0.8", "remote")
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


if __name__ == "__main__":
    unittest.main()
