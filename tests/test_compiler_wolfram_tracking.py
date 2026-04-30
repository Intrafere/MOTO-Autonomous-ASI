import unittest

from backend.autonomous.memory.paper_model_tracker import PaperModelTracker
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.shared.models import CompilerSubmission


def _submission_with_wolfram_calls() -> CompilerSubmission:
    return CompilerSubmission(
        submission_id="sub-wolfram",
        mode="construction",
        content="content",
        operation="full_content",
        old_string="",
        new_string="content",
        reasoning="accepted construction",
        metadata={
            "wolfram_calls": [
                {"query": "2+2", "result": "4"},
                {"query": "integrate x", "result": "x^2/2"},
            ]
        },
    )


class CompilerWolframTrackingTests(unittest.TestCase):
    def test_manual_mode_tracks_accepted_wolfram_calls(self) -> None:
        coordinator = CompilerCoordinator()
        tracker = PaperModelTracker(user_prompt="prompt", paper_title="paper")
        coordinator._paper_model_tracker = tracker

        coordinator._track_submission_wolfram_calls(_submission_with_wolfram_calls())

        self.assertEqual(tracker.get_wolfram_call_count(), 2)

    def test_autonomous_mode_tracks_on_current_paper_tracker(self) -> None:
        coordinator = CompilerCoordinator()
        coordinator.enable_autonomous_mode()
        manual_tracker = PaperModelTracker(user_prompt="manual", paper_title="manual")
        autonomous_tracker = PaperModelTracker(user_prompt="auto", paper_title="auto")
        coordinator._paper_model_tracker = manual_tracker
        coordinator._current_paper_tracker = autonomous_tracker

        coordinator._track_submission_wolfram_calls(_submission_with_wolfram_calls())

        self.assertEqual(manual_tracker.get_wolfram_call_count(), 0)
        self.assertEqual(autonomous_tracker.get_wolfram_call_count(), 2)


if __name__ == "__main__":
    unittest.main()
