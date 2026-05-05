import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.shared.lean4_client import (
    Lean4Client,
    _deduplicate_leading_import,
    _strip_markdown_fences,
)


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

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int) -> tuple[int, str, str]:
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

            async def fake_run_process(args: list[str], *, cwd: Path, timeout: int) -> tuple[int, str, str]:
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
                ],
            )


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
