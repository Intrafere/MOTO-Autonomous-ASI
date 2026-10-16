import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.shared import lean4_client, lean_workspace_paths
from backend.shared.lean4_client import (
    LEAN_WORKSPACE_PATH_TOO_LONG,
    LEAN_WORKSPACE_SETUP_FAILED,
    Lean4Client,
    Lean4Result,
    LeanWorkspaceRepairRequiredError,
    _deduplicate_leading_import,
    _strip_markdown_fences,
    raise_for_lean_workspace_failure,
    recover_lean_workspace,
)
from backend.shared.lean_workspace_paths import MAX_SAFE_WORKSPACE_PATH


class Lean4ClientWorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_fetch_retries_after_removing_failed_ltar_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            mathlib_pkg = workspace / ".lake" / "packages" / "mathlib"
            mathlib_pkg.mkdir(parents=True)
            lake_path = root / "lake.exe"
            lake_path.write_text("", encoding="utf-8")

            failed_archive = root / ".cache" / "mathlib" / "bad.ltar"
            failed_archive.parent.mkdir(parents=True)
            failed_archive.write_bytes(b"partial archive")

            client = Lean4Client(lean_path=str(root / "lean.exe"), workspace_dir=str(workspace))
            calls: list[list[str]] = []

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                if args[1:] == ["update"]:
                    return 0, "updated", ""
                if len(calls) == 2:
                    return (
                        101,
                        "",
                        f"Decompression error: leantar exited with code 101 ({failed_archive})",
                    )
                return 0, "cache ok", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]

            self.assertTrue(await client.ensure_workspace())
            self.assertFalse(failed_archive.exists())
            self.assertEqual(
                [call[1:] for call in calls],
                [
                    ["update"],
                    ["exe", "cache", "get"],
                    ["exe", "cache", "get"],
                    ["env", str(root / "lean.exe"), "MOTOProofWorkspace.lean"],
                    ["exe", "cache", "clean"],
                ],
            )

    async def test_workspace_bootstrap_is_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            (workspace / ".lake" / "packages" / "mathlib").mkdir(parents=True)
            lake_path = root / "lake.exe"
            lake_path.write_text("", encoding="utf-8")

            client = Lean4Client(lean_path=str(root / "lean.exe"), workspace_dir=str(workspace))
            calls: list[list[str]] = []

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                await asyncio.sleep(0.01)
                return 0, "ok", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]

            results = await asyncio.gather(client.ensure_workspace(), client.ensure_workspace())

            self.assertEqual(results, [True, True])
            self.assertEqual(
                [call[1:] for call in calls],
                [
                    ["update"],
                    ["exe", "cache", "get"],
                    ["env", str(root / "lean.exe"), "MOTOProofWorkspace.lean"],
                    ["exe", "cache", "clean"],
                ],
            )

    async def test_workspace_status_is_observational_and_bootstrap_is_single_flight(self) -> None:
        client = Lean4Client(lean_path="", workspace_dir=tempfile.gettempdir())
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def fake_ensure() -> bool:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            client._workspace_ready = True
            return True

        client._ensure_workspace_locked = fake_ensure  # type: ignore[method-assign]
        self.assertEqual(client.get_workspace_status()["state"], "not_started")
        first = client.start_workspace_bootstrap()
        second = client.start_workspace_bootstrap()
        self.assertIs(first, second)
        await started.wait()
        self.assertEqual(client.get_workspace_status()["state"], "starting")
        release.set()
        self.assertEqual(await asyncio.gather(first, second), [True, True])
        self.assertEqual(calls, 1)
        self.assertEqual(client.get_workspace_status()["state"], "ready")

    async def test_cancelled_waiter_does_not_cancel_shared_bootstrap(self) -> None:
        client = Lean4Client(lean_path="", workspace_dir=tempfile.gettempdir())
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_ensure() -> bool:
            started.set()
            await release.wait()
            client._workspace_ready = True
            return True

        client._ensure_workspace_locked = fake_ensure  # type: ignore[method-assign]
        waiter = asyncio.create_task(client.await_workspace_ready())
        await started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(client.start_workspace_bootstrap().cancelled())
        release.set()
        self.assertTrue(await client.await_workspace_ready())


class Lean4WorkspaceRepairTests(unittest.IsolatedAsyncioTestCase):
    def _client(self, root: Path, workspace: Path) -> Lean4Client:
        (root / "lake.exe").write_text("", encoding="utf-8")
        return Lean4Client(lean_path=str(root / "lean.exe"), workspace_dir=str(workspace))

    async def test_long_windows_workspace_path_fails_before_any_lake_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / ("w" * (MAX_SAFE_WORKSPACE_PATH + 1))
            client = self._client(root, workspace)
            calls: list[list[str]] = []

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                return 0, "", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]
            with mock.patch.object(lean_workspace_paths, "is_windows_platform", return_value=True):
                self.assertFalse(await client.ensure_workspace())
                with self.assertRaises(LeanWorkspaceRepairRequiredError) as raised:
                    client.require_workspace()

            self.assertEqual(calls, [])
            self.assertFalse(workspace.exists())
            self.assertEqual(raised.exception.failure_kind, LEAN_WORKSPACE_PATH_TOO_LONG)
            self.assertIn("folder path is too long", client.get_workspace_status()["error"])

    async def test_windows_path_error_is_diagnosed_without_deleting_cache_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "ws"
            mathlib_sources = workspace / ".lake" / "packages" / "mathlib" / "Mathlib"
            mathlib_sources.mkdir(parents=True)
            # Make the module's build output land 5 characters past Windows MAX_PATH.
            module_length = 200 - len(str(workspace.resolve()))
            (mathlib_sources / f"{'M' * (module_length - len('Mathlib/'))}.lean").write_text(
                "", encoding="utf-8"
            )
            archive = root / ".cache" / "mathlib" / "intact.ltar"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"valid archive")
            client = self._client(root, workspace)
            calls: list[list[str]] = []

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                if args[1:] == ["update"]:
                    return 0, "updated", ""
                return 1, "", f"({archive}) The system cannot find the path specified. (os error 3)"

            client._run_process = fake_run_process  # type: ignore[method-assign]
            with mock.patch.object(lean_workspace_paths, "is_windows_platform", return_value=True):
                self.assertFalse(await client.ensure_workspace())

            self.assertEqual([call[1:] for call in calls], [["update"], ["exe", "cache", "get"]])
            self.assertTrue(archive.exists())
            self.assertEqual(
                client.workspace_repair_required_error().failure_kind,
                LEAN_WORKSPACE_PATH_TOO_LONG,
            )

    async def test_require_workspace_only_raises_for_a_known_failure(self) -> None:
        client = Lean4Client(lean_path="", workspace_dir=tempfile.gettempdir())
        calls = 0

        async def fake_ensure() -> bool:
            nonlocal calls
            calls += 1
            return False

        client._ensure_workspace_locked = fake_ensure  # type: ignore[method-assign]
        client.require_workspace()
        self.assertEqual(calls, 0)

        self.assertFalse(await client.await_workspace_ready())
        with self.assertRaises(LeanWorkspaceRepairRequiredError) as raised:
            client.require_workspace()
        self.assertEqual(calls, 1)
        self.assertEqual(raised.exception.failure_kind, LEAN_WORKSPACE_SETUP_FAILED)

    async def test_round_guard_gives_a_failed_setup_one_fresh_attempt(self) -> None:
        client = Lean4Client(lean_path="", workspace_dir=tempfile.gettempdir())
        outcomes = [False, True]
        calls = 0

        async def fake_ensure() -> bool:
            nonlocal calls
            calls += 1
            ready = outcomes.pop(0)
            client._workspace_ready = ready
            return ready

        client._ensure_workspace_locked = fake_ensure  # type: ignore[method-assign]
        self.assertFalse(await client.await_workspace_ready())

        await recover_lean_workspace(client)

        self.assertEqual(calls, 2)
        client.require_workspace()

    async def test_round_guard_never_retries_a_too_long_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            client = self._client(root, root / ("w" * (MAX_SAFE_WORKSPACE_PATH + 1)))
            ensure = mock.AsyncMock(return_value=True)
            client._ensure_workspace_locked = ensure  # type: ignore[method-assign]
            with mock.patch.object(lean_workspace_paths, "is_windows_platform", return_value=True):
                with self.assertRaises(LeanWorkspaceRepairRequiredError):
                    await recover_lean_workspace(client)
            ensure.assert_not_awaited()

    def _installed_workspace(self, root: Path) -> Path:
        workspace = root / "workspace"
        (workspace / ".lake" / "packages" / "mathlib" / ".git").mkdir(parents=True)
        (workspace / "lake-manifest.json").write_text('{"packages": []}', encoding="utf-8")
        return workspace

    async def test_installed_mathlib_is_kept_instead_of_updated_to_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            client = self._client(root, self._installed_workspace(root))
            calls: list[list[str]] = []

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                return 0, "", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]
            with mock.patch.object(lean4_client.shutil, "which", side_effect=lambda name: name if name == "git" else None):
                self.assertTrue(await client.ensure_workspace())

            self.assertEqual(
                [call[1:] for call in calls],
                [
                    ["status", "--porcelain", "--untracked-files=normal"],
                    ["exe", "cache", "get"],
                    ["env", str(root / "lean.exe"), "MOTOProofWorkspace.lean"],
                    ["exe", "cache", "clean"],
                ],
            )

    async def test_interrupted_checkout_is_restored_without_redownloading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            client = self._client(root, self._installed_workspace(root))
            calls: list[list[str]] = []
            statuses = [" M Mathlib.lean\n?? Mathlib/New.lean\n", ""]

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                if args[1] == "status":
                    return 0, statuses.pop(0), ""
                return 0, "", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]
            with mock.patch.object(lean4_client.shutil, "which", side_effect=lambda name: name if name == "git" else None):
                self.assertTrue(await client.ensure_workspace())

            self.assertEqual(
                [call[1:] for call in calls],
                [
                    ["status", "--porcelain", "--untracked-files=normal"],
                    ["reset", "--hard", "--quiet", "HEAD"],
                    ["clean", "-fd", "--quiet"],
                    ["status", "--porcelain", "--untracked-files=normal"],
                    ["exe", "cache", "get"],
                    ["env", str(root / "lean.exe"), "MOTOProofWorkspace.lean"],
                    ["exe", "cache", "clean"],
                ],
            )

    async def test_unrestorable_checkout_is_wiped_and_reinstalled_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = self._installed_workspace(root)
            client = self._client(root, workspace)
            calls: list[list[str]] = []
            disk_full = True

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str, str]:
                calls.append(args)
                if args[1] == "status":
                    return 0, "?? Mathlib/New.lean\n", ""
                if args[1] == "reset":
                    return 128, "", "fatal: Unable to create '.git/index.lock': File exists."
                if args[1:] == ["update"]:
                    if disk_full:
                        return 1, "", "error: No space left on device"
                    (workspace / ".lake" / "packages" / "mathlib").mkdir(parents=True)
                return 0, "", ""

            client._run_process = fake_run_process  # type: ignore[method-assign]
            with mock.patch.object(lean4_client.shutil, "which", side_effect=lambda name: name if name == "git" else None):
                self.assertFalse(await client.ensure_workspace())
                self.assertFalse((workspace / ".lake").exists())
                with self.assertRaises(LeanWorkspaceRepairRequiredError):
                    client.require_workspace()

                # Space freed mid-session: the per-round retry reinstalls without a restart.
                disk_full = False
                calls.clear()
                await recover_lean_workspace(client)

            self.assertEqual(
                [call[1:] for call in calls],
                [
                    ["update"],
                    ["exe", "cache", "get"],
                    ["env", str(root / "lean.exe"), "MOTOProofWorkspace.lean"],
                    ["exe", "cache", "clean"],
                ],
            )
            client.require_workspace()

    def test_interrupted_checkout_output_counts_as_stale_lake_state(self) -> None:
        output = (
            "info: mathlib: checking out revision 'e91b128'\n"
            "error: Your local changes to the following files would be overwritten by checkout:\n"
            "error: external command 'git' exited with code 1"
        )
        self.assertTrue(Lean4Client._is_stale_lake_state(output))
        self.assertFalse(Lean4Client._is_stale_lake_state("error: could not resolve host github.com"))

    async def test_mathlib_downloads_go_to_a_moto_owned_cache_that_is_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            client = Lean4Client(lean_path="", workspace_dir=str(workspace))
            with mock.patch.dict(lean4_client.os.environ, {}, clear=False):
                lean4_client.os.environ.pop("MATHLIB_CACHE_DIR", None)
                self.assertEqual(client.mathlib_cache_dir, workspace.resolve() / ".moto_mathlib_cache")
                self.assertTrue(client._owns_mathlib_cache_dir())

            operator_dir = Path(tmpdir) / "operator_cache"
            run_process = mock.AsyncMock(return_value=(0, "", ""))
            client._run_process = run_process  # type: ignore[method-assign]
            with mock.patch.dict(lean4_client.os.environ, {"MATHLIB_CACHE_DIR": str(operator_dir)}):
                self.assertEqual(client.mathlib_cache_dir, operator_dir)
                await client._prune_mathlib_cache("lake")
            run_process.assert_not_awaited()

    def test_workspace_failure_result_becomes_the_typed_repair_error(self) -> None:
        raise_for_lean_workspace_failure(Lean4Result(success=False, error_output="error: unsolved goals"))
        with self.assertRaises(LeanWorkspaceRepairRequiredError) as raised:
            raise_for_lean_workspace_failure(
                Lean4Result(success=False, error_output="LEAN 4 WORKSPACE ERROR: Mathlib cache failed.")
            )
        self.assertTrue(str(raised.exception).startswith("LEAN 4 WORKSPACE ERROR"))
        self.assertEqual(raised.exception.to_payload()["reason"], "lean_workspace_repair_required")


class Lean4ExtractionTests(unittest.IsolatedAsyncioTestCase):
    def _client(self) -> Lean4Client:
        return Lean4Client(lean_path="", workspace_dir=tempfile.gettempdir())

    def test_strip_markdown_fences_removes_lean_fence(self) -> None:
        fence = "`" * 3
        code = f"{fence}lean\nimport Mathlib\ntheorem t : 1 = 1 := rfl\n{fence}"
        result = _strip_markdown_fences(code)
        self.assertNotIn(fence, result)
        self.assertIn("import Mathlib", result)
        self.assertIn("theorem t : 1 = 1 := rfl", result)

    def test_strip_markdown_fences_noop_when_no_fences(self) -> None:
        code = "import Mathlib\ntheorem t : 1 = 1 := rfl"
        self.assertEqual(_strip_markdown_fences(code), code)

    def test_deduplicate_leading_import_collapses_duplicates(self) -> None:
        code = "import Mathlib\nimport Mathlib\ntheorem t : 1 = 1 := rfl"
        result = _deduplicate_leading_import(code)
        self.assertEqual(result.count("import Mathlib"), 1)
        self.assertIn("theorem t", result)

    def test_prepare_lean_code_strips_fences_and_adds_import(self) -> None:
        client = self._client()
        fence = "`" * 3
        code = f"{fence}\ntheorem t : 1 = 1 := rfl\n{fence}"
        prepared = client._prepare_lean_code(code)
        self.assertNotIn(fence, prepared)
        self.assertTrue(prepared.startswith("import Mathlib"))
        self.assertIn("theorem t : 1 = 1 := rfl", prepared)

    def test_has_no_goals_diagnostic_detects_error(self) -> None:
        self.assertTrue(
            Lean4Client._has_no_goals_diagnostic(
                "file.lean:43:9: error: No goals to be solved"
            )
        )
        self.assertFalse(
            Lean4Client._has_no_goals_diagnostic(
                "file.lean:10:3: error: unknown identifier 'foo'"
            )
        )

    def test_annotate_no_goals_hint_prepends_hint_once(self) -> None:
        raw = "file.lean:43:9: error: No goals to be solved"
        annotated = Lean4Client._annotate_no_goals_hint(raw)
        self.assertIn("HINT:", annotated)
        self.assertIn("no goals to be solved", annotated.lower())
        annotated_twice = Lean4Client._annotate_no_goals_hint(annotated)
        self.assertEqual(annotated_twice.count("HINT:"), 1)

    def test_annotate_no_goals_hint_skips_unrelated_error(self) -> None:
        raw = "file.lean:10:3: error: unknown identifier 'foo'"
        self.assertEqual(Lean4Client._annotate_no_goals_hint(raw), raw)


if __name__ == "__main__":
    unittest.main()
