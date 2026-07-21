import tempfile
import unittest
from pathlib import Path

from backend.compiler.memory.paper_memory import (
    APPENDIX_EMPTY_PLACEHOLDER,
    ABSTRACT_PLACEHOLDER,
    AI_SELF_REVIEW_SECTION_HEADER,
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

    async def test_appendix_skips_duplicate_proof_id(self) -> None:
        old_paper_file = system_config.compiler_paper_file
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.compiler_paper_file = str(Path(tmpdir) / "paper.txt")
                memory = PaperMemory()
                await memory.initialize()
                await memory.initialize_with_placeholders("II. Body\n\nBody text.")

                entry = (
                    "### Theorem A\n"
                    "Proof ID: proof_duplicate\n"
                    "Verified with Lean 4.\n"
                    "```lean\n"
                    "theorem a : True := by trivial\n"
                    "```"
                )

                self.assertTrue(await memory.append_to_theorems_appendix(entry))
                self.assertTrue(await memory.append_to_theorems_appendix(entry))

                paper = await memory.get_paper()
                self.assertEqual(paper.count("Proof ID: proof_duplicate"), 1)
                self.assertEqual(paper.count("theorem a : True"), 1)
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

    async def test_repairs_preserve_appendix_then_self_review_order(self) -> None:
        for repair_method in ("ensure_placeholders_exist", "ensure_markers_intact"):
            old_paper_file = system_config.compiler_paper_file
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    system_config.compiler_paper_file = str(
                        Path(tmpdir) / f"{repair_method}.txt"
                    )
                    memory = PaperMemory()
                    await memory.initialize()
                    theorem = (
                        "### Verified Theorem\n"
                        "Proof ID: proof-order\n"
                        "```lean\n"
                        "theorem ordered : True := by trivial\n"
                        "```"
                    )
                    self_review = (
                        f"{AI_SELF_REVIEW_SECTION_HEADER}\n\n"
                        "The limitations section remains after verified proofs."
                    )
                    abstract = "Abstract\n\n" + ("Substantive abstract content. " * 20)
                    await memory.update_paper(
                        f"{ABSTRACT_PLACEHOLDER}\n\n{abstract}\n\n"
                        f"{INTRO_PLACEHOLDER}\n\n"
                        "II. Body\n\nBody text.\n\n"
                        f"{CONCLUSION_PLACEHOLDER}\n\n"
                        f"{THEOREMS_APPENDIX_START}\n"
                        f"{theorem}\n"
                        f"{THEOREMS_APPENDIX_END}\n\n"
                        f"{self_review}\n"
                    )

                    self.assertTrue(await getattr(memory, repair_method)())
                    paper = await memory.get_paper()

                    self.assertEqual(paper.count("Proof ID: proof-order"), 1)
                    self.assertEqual(
                        paper.count(AI_SELF_REVIEW_SECTION_HEADER), 1
                    )
                    self.assertLess(paper.index("II. Body"), paper.index(theorem))
                    self.assertLess(paper.index(theorem), paper.index(self_review))
                    self.assertLess(paper.index(self_review), paper.index(PAPER_ANCHOR))
                finally:
                    system_config.compiler_paper_file = old_paper_file

    async def test_replacing_misordered_self_review_preserves_appendix(self) -> None:
        old_paper_file = system_config.compiler_paper_file
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system_config.compiler_paper_file = str(Path(tmpdir) / "paper.txt")
                memory = PaperMemory()
                await memory.initialize()
                theorem = (
                    "Proof ID: proof-preserved\n"
                    "```lean\n"
                    "theorem preserved : True := by trivial\n"
                    "```"
                )
                await memory.update_paper(
                    "II. Body\n\n"
                    f"{AI_SELF_REVIEW_SECTION_HEADER}\n\nOld review.\n\n"
                    f"{THEOREMS_APPENDIX_START}\n{theorem}\n"
                    f"{THEOREMS_APPENDIX_END}\n\n{PAPER_ANCHOR}"
                )

                self.assertTrue(
                    await memory.append_self_review_section("Replacement review.")
                )
                paper = await memory.get_paper()

                self.assertEqual(paper.count("Proof ID: proof-preserved"), 1)
                self.assertEqual(paper.count("theorem preserved : True"), 1)
                self.assertEqual(paper.count(AI_SELF_REVIEW_SECTION_HEADER), 1)
                self.assertIn("Replacement review.", paper)
                self.assertNotIn("Old review.", paper)
                self.assertLess(
                    paper.index(THEOREMS_APPENDIX_END),
                    paper.index(AI_SELF_REVIEW_SECTION_HEADER),
                )
            finally:
                system_config.compiler_paper_file = old_paper_file


if __name__ == "__main__":
    unittest.main()
