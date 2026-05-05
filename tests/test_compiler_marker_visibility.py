import unittest
import tempfile
from pathlib import Path

from backend.compiler.core import compiler_coordinator as coordinator_module
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.compiler.agents.high_context_submitter import (
    _strip_paper_markers_for_llm as strip_for_high_context,
)
from backend.compiler.agents.high_param_submitter import (
    _strip_paper_markers_for_llm as strip_for_high_param,
)
from backend.compiler.memory.paper_memory import (
    APPENDIX_EMPTY_PLACEHOLDER,
    CONCLUSION_PLACEHOLDER,
    PAPER_ANCHOR,
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
    paper_memory,
)
from backend.compiler.validation.compiler_validator import CompilerValidator
from backend.shared.models import CompilerSubmission, CompilerValidationResult


class CompilerMarkerVisibilityTests(unittest.TestCase):
    def test_submitter_paper_view_preserves_appendix_markers_for_exact_matching(self) -> None:
        paper = (
            "Body text.\n\n"
            f"{THEOREMS_APPENDIX_START}\n"
            "[Theorems appendix - verified Lean 4 theorems not placed inline will appear here]\n"
            f"{THEOREMS_APPENDIX_END}\n\n"
            f"{PAPER_ANCHOR}"
        )

        for strip_for_llm in (strip_for_high_context, strip_for_high_param):
            with self.subTest(strip_for_llm=strip_for_llm.__module__):
                visible_paper = strip_for_llm(paper)

                self.assertIn(THEOREMS_APPENDIX_START, visible_paper)
                self.assertIn(THEOREMS_APPENDIX_END, visible_paper)
                self.assertIn(PAPER_ANCHOR, visible_paper)

    def test_replace_old_string_with_appendix_marker_suffix_is_trimmed_safely(self) -> None:
        validator = CompilerValidator(model_name="test-model", user_prompt="Write.")
        old_section = "\\section{Conclusion}\n\nOld conclusion text."
        paper = (
            f"{old_section}\n\n"
            f"{THEOREMS_APPENDIX_START}\n"
            f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
            f"{THEOREMS_APPENDIX_END}\n\n"
            f"{PAPER_ANCHOR}"
        )
        submission = CompilerSubmission(
            submission_id="sub-marker-trim",
            mode="construction",
            content="\\section{Conclusion}\n\nNew conclusion text.",
            operation="replace",
            old_string=(
                f"{old_section}\n\n"
                f"{THEOREMS_APPENDIX_START}\n"
                f"{APPENDIX_EMPTY_PLACEHOLDER}"
            ),
            new_string="\\section{Conclusion}\n\nNew conclusion text.",
            reasoning="Replace conclusion.",
        )

        result = validator._pre_validate_exact_string_match(submission, paper, "Outline")

        self.assertIsNone(result)
        self.assertEqual(submission.old_string, old_section)

    def test_delete_old_string_crossing_appendix_marker_is_rejected(self) -> None:
        validator = CompilerValidator(model_name="test-model", user_prompt="Write.")
        paper = (
            "Conclusion text.\n\n"
            f"{THEOREMS_APPENDIX_START}\n"
            f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
            f"{THEOREMS_APPENDIX_END}\n\n"
            f"{PAPER_ANCHOR}"
        )
        submission = CompilerSubmission(
            submission_id="sub-marker-delete",
            mode="construction",
            content="",
            operation="delete",
            old_string=f"Conclusion text.\n\n{THEOREMS_APPENDIX_START}",
            new_string="",
            reasoning="Delete stale conclusion.",
        )

        result = validator._pre_validate_exact_string_match(submission, paper, "Outline")

        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "reject")
        self.assertIn("PROTECTED_MARKER_BOUNDARY", result.reasoning)


class CompilerCoordinatorMarkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_conclusion_phase_without_placeholder_applies_validated_edit(self) -> None:
        old_path = paper_memory.file_path
        old_initialized = paper_memory._initialized
        old_rechunk_callback = paper_memory.rechunk_callback
        old_add_acceptance = coordinator_module.compiler_rejection_log.add_acceptance

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                paper_memory.file_path = Path(tmpdir) / "paper.txt"
                paper_memory._initialized = True
                paper_memory.rechunk_callback = None

                old_conclusion = "\\section{Conclusion}\n\nOld conclusion."
                new_conclusion = "\\section{Conclusion}\n\nNew conclusion."
                await paper_memory.update_paper(
                    f"Body text.\n\n"
                    f"{old_conclusion}\n\n"
                    f"{THEOREMS_APPENDIX_START}\n"
                    f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
                    f"{THEOREMS_APPENDIX_END}\n\n"
                    f"{PAPER_ANCHOR}"
                )

                submission = CompilerSubmission(
                    submission_id="sub-conclusion-existing",
                    mode="construction",
                    content=new_conclusion,
                    operation="replace",
                    old_string=old_conclusion,
                    new_string=new_conclusion,
                    reasoning="Refresh existing conclusion.",
                    section_complete=False,
                )

                class FakeSubmitter:
                    async def submit_construction(self, **_kwargs):
                        return submission

                class FakeValidator:
                    async def validate_submission(self, *_args, **_kwargs):
                        return CompilerValidationResult(
                            submission_id=submission.submission_id,
                            decision="accept",
                            reasoning="Accepted.",
                            coherence_check=True,
                            rigor_check=True,
                            placement_check=True,
                        )

                async def fake_add_acceptance(*_args, **_kwargs):
                    return None

                coordinator_module.compiler_rejection_log.add_acceptance = fake_add_acceptance

                coordinator = CompilerCoordinator()
                coordinator.autonomous_mode = True
                coordinator.autonomous_section_phase = "conclusion"
                coordinator.high_context_submitter = FakeSubmitter()
                coordinator.validator = FakeValidator()

                accepted, rejection_reason = await coordinator._submit_and_validate_construction()

                paper = await paper_memory.get_paper()
                self.assertTrue(accepted)
                self.assertIsNone(rejection_reason)
                self.assertIn(new_conclusion, paper)
                self.assertNotIn(old_conclusion, paper)
                self.assertIn(THEOREMS_APPENDIX_START, paper)
                self.assertIn(THEOREMS_APPENDIX_END, paper)
            finally:
                paper_memory.file_path = old_path
                paper_memory._initialized = old_initialized
                paper_memory.rechunk_callback = old_rechunk_callback
                coordinator_module.compiler_rejection_log.add_acceptance = old_add_acceptance

    async def test_phase_full_content_with_placeholder_replaces_placeholder_not_whole_paper(self) -> None:
        old_path = paper_memory.file_path
        old_initialized = paper_memory._initialized
        old_rechunk_callback = paper_memory.rechunk_callback
        old_add_acceptance = coordinator_module.compiler_rejection_log.add_acceptance

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                paper_memory.file_path = Path(tmpdir) / "paper.txt"
                paper_memory._initialized = True
                paper_memory.rechunk_callback = None

                new_conclusion = "\\section{Conclusion}\n\nNew conclusion."
                await paper_memory.update_paper(
                    "Body text.\n\n"
                    f"{CONCLUSION_PLACEHOLDER}\n\n"
                    f"{THEOREMS_APPENDIX_START}\n"
                    f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
                    f"{THEOREMS_APPENDIX_END}\n\n"
                    f"{PAPER_ANCHOR}"
                )

                submission = CompilerSubmission(
                    submission_id="sub-conclusion-full-content",
                    mode="construction",
                    content=new_conclusion,
                    operation="full_content",
                    old_string="Body text.",
                    new_string=new_conclusion,
                    reasoning="Write conclusion.",
                    section_complete=False,
                )

                class FakeSubmitter:
                    async def submit_construction(self, **_kwargs):
                        return submission

                class FakeValidator:
                    async def validate_submission(self, *_args, **_kwargs):
                        return CompilerValidationResult(
                            submission_id=submission.submission_id,
                            decision="accept",
                            reasoning="Accepted.",
                            coherence_check=True,
                            rigor_check=True,
                            placement_check=True,
                        )

                async def fake_add_acceptance(*_args, **_kwargs):
                    return None

                coordinator_module.compiler_rejection_log.add_acceptance = fake_add_acceptance

                coordinator = CompilerCoordinator()
                coordinator.autonomous_mode = True
                coordinator.autonomous_section_phase = "conclusion"
                coordinator.high_context_submitter = FakeSubmitter()
                coordinator.validator = FakeValidator()

                accepted, rejection_reason = await coordinator._submit_and_validate_construction()

                paper = await paper_memory.get_paper()
                self.assertTrue(accepted)
                self.assertIsNone(rejection_reason)
                self.assertIn("Body text.", paper)
                self.assertIn(new_conclusion, paper)
                self.assertNotIn(CONCLUSION_PLACEHOLDER, paper)
                self.assertIn(THEOREMS_APPENDIX_START, paper)
                self.assertIn(THEOREMS_APPENDIX_END, paper)
            finally:
                paper_memory.file_path = old_path
                paper_memory._initialized = old_initialized
                paper_memory.rechunk_callback = old_rechunk_callback
                coordinator_module.compiler_rejection_log.add_acceptance = old_add_acceptance


if __name__ == "__main__":
    unittest.main()
