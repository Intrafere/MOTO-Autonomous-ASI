import tempfile
import unittest
from pathlib import Path

from backend.compiler.memory.paper_memory import (
    APPENDIX_EMPTY_PLACEHOLDER,
    ABSTRACT_PLACEHOLDER,
    CONCLUSION_PLACEHOLDER,
    INTRO_PLACEHOLDER,
    PAPER_ANCHOR,
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
    PaperMemory,
)
from backend.shared.config import system_config


class PaperMemoryAppendixTests(unittest.IsolatedAsyncioTestCase):
    async def test_appendix_entries_replace_placeholder_and_append_in_order(self) -> None:
        old_paper_file = system_config.compiler_paper_file
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.compiler_paper_file = str(Path(tmpdir) / "paper.txt")
                memory = PaperMemory()
                await memory.initialize()
                await memory.initialize_with_placeholders("II. Body\n\nBody text.")

                first = "### Theorem A\nVerified with Lean 4.\n```lean\ntheorem a : True := by trivial\n```"
                second = "### Theorem B\nVerified with Lean 4.\n```lean\ntheorem b : True := by trivial\n```"

                self.assertTrue(await memory.append_to_theorems_appendix(first))
                self.assertTrue(await memory.append_to_theorems_appendix(second))

                paper = await memory.get_paper()
                self.assertIn(THEOREMS_APPENDIX_START, paper)
                self.assertIn(THEOREMS_APPENDIX_END, paper)
                self.assertIn(first, paper)
                self.assertIn(second, paper)
                self.assertNotIn(APPENDIX_EMPTY_PLACEHOLDER, paper)
                self.assertLess(paper.index(first), paper.index(second))
                self.assertTrue(paper.rstrip().endswith(PAPER_ANCHOR))
            finally:
                system_config.compiler_paper_file = old_paper_file

    async def test_latex_conclusion_content_removes_stale_placeholder(self) -> None:
        old_paper_file = system_config.compiler_paper_file
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.compiler_paper_file = str(Path(tmpdir) / "paper.txt")
                memory = PaperMemory()
                await memory.initialize()

                conclusion = "This is real conclusion content. " * 20
                await memory.update_paper(
                    f"{ABSTRACT_PLACEHOLDER}\n\n"
                    f"{INTRO_PLACEHOLDER}\n\n"
                    "\\section{Preliminaries}\n\nBody text.\n\n"
                    f"\\section{{Conclusion}}\n\n{conclusion}\n\n"
                    f"{CONCLUSION_PLACEHOLDER}\n\n"
                    f"{THEOREMS_APPENDIX_START}\n"
                    f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
                    f"{THEOREMS_APPENDIX_END}\n\n"
                    f"{PAPER_ANCHOR}"
                )

                self.assertTrue(await memory.ensure_placeholders_exist())

                paper = await memory.get_paper()
                self.assertIn("\\section{Conclusion}", paper)
                self.assertNotIn(CONCLUSION_PLACEHOLDER, paper)
                self.assertIn(ABSTRACT_PLACEHOLDER, paper)
                self.assertIn(INTRO_PLACEHOLDER, paper)
            finally:
                system_config.compiler_paper_file = old_paper_file


if __name__ == "__main__":
    unittest.main()
