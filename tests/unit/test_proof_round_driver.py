import unittest
from types import SimpleNamespace

from backend.autonomous.core.proof_round_driver import (
    AutomaticMultiRoundPolicy,
    ContinuousPruningPolicy,
    OneRoundPolicy,
    ProofRoundDriver,
    summarize_round_result,
)


class ProofRoundPolicyTests(unittest.TestCase):
    def test_policies_expose_explicit_round_and_reservation_semantics(self):
        one_round = OneRoundPolicy()
        automatic = AutomaticMultiRoundPolicy(max_rounds=4)
        continuous = ContinuousPruningPolicy()

        self.assertEqual(one_round.max_rounds, 1)
        self.assertFalse(one_round.holds_source_reservation)
        self.assertEqual(one_round.trigger_for_round("retry", 2), "retry")
        self.assertEqual(automatic.max_rounds, 4)
        self.assertTrue(automatic.holds_source_reservation)
        self.assertEqual(
            [automatic.trigger_for_round("automatic", index) for index in range(1, 4)],
            ["automatic", "automatic_round_2", "automatic_round_3"],
        )
        self.assertIsNone(continuous.max_rounds)
        self.assertTrue(continuous.holds_source_reservation)

    def test_round_summary_is_bounded(self):
        results = [
            SimpleNamespace(
                success=index % 2 == 0,
                theorem_statement=("theorem-" + str(index)) * 100,
            )
            for index in range(7)
        ]
        summary = summarize_round_result(
            2,
            SimpleNamespace(
                verified_count=3,
                total_candidates=7,
                novel_count=2,
                results=results,
            ),
        )

        self.assertIn("Round 2: 3/7 candidates verified, 2 novel.", summary)
        self.assertEqual(summary.count("\n- "), 5)
        self.assertNotIn("theorem-6", summary)


class ProofRoundDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_round_holds_one_reservation_and_stops_on_no_candidates(self):
        calls = []
        reserves = []
        releases = []
        totals = [1, 1, 0, 1]

        async def reserve(source_type, source_id):
            reserves.append((source_type, source_id))
            return "owner-token"

        async def release(source_type, source_id, token):
            releases.append((source_type, source_id, token))

        async def execute(index, trigger, prior_results, token):
            calls.append((index, trigger, prior_results, token))
            total = totals[index - 1]
            return "completed", SimpleNamespace(
                verified_count=total,
                total_candidates=total,
                novel_count=total,
                results=[],
                had_error=False,
            )

        driver = ProofRoundDriver(
            policy=AutomaticMultiRoundPolicy(max_rounds=4),
            source_type="paper",
            source_id="paper-1",
            base_trigger="automatic",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=reserve,
            release_source=release,
        )

        self.assertEqual(await driver.run(), "complete")
        self.assertEqual(len(calls), 3)
        self.assertEqual(reserves, [("paper", "paper-1")])
        self.assertEqual(releases, [("paper", "paper-1", "owner-token")])
        self.assertTrue(all(call[3] == "owner-token" for call in calls))
        self.assertEqual(calls[2][1], "automatic_round_3")
        self.assertIn("Round 1:", calls[1][2])
        self.assertIn("Round 2:", calls[2][2])

    async def test_one_round_does_not_reserve(self):
        calls = []

        async def execute(index, trigger, prior_results, token):
            calls.append((index, trigger, prior_results, token))
            return "completed", SimpleNamespace(
                verified_count=1,
                total_candidates=1,
                novel_count=0,
                results=[],
                had_error=False,
            )

        async def unexpected(*_args):
            self.fail("one-round policy must not reserve or release")

        driver = ProofRoundDriver(
            policy=OneRoundPolicy(),
            source_type="paper",
            source_id="paper-1",
            base_trigger="retry",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=unexpected,
            release_source=unexpected,
        )

        self.assertEqual(await driver.run(), "complete")
        self.assertEqual(calls, [(1, "retry", "", "")])

    async def test_only_valid_no_candidates_outcome_stops_follow_up_rounds(self):
        calls = []

        async def reserve(*_args):
            return "owner-token"

        async def release(*_args):
            return None

        async def execute(index, trigger, prior_results, token):
            calls.append(index)
            return "no_candidates_skipped", None

        driver = ProofRoundDriver(
            policy=AutomaticMultiRoundPolicy(max_rounds=4),
            source_type="brainstorm",
            source_id="topic-1",
            base_trigger="automatic",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=reserve,
            release_source=release,
        )

        self.assertEqual(await driver.run(), "complete")
        self.assertEqual(calls, [1])

    async def test_fatal_stop_outcome_is_preserved_for_run_lifecycle(self):
        async def reserve(*_args):
            return "owner-token"

        async def release(*_args):
            return None

        async def execute(*_args):
            return "fatal_stop", None

        driver = ProofRoundDriver(
            policy=OneRoundPolicy(),
            source_type="paper",
            source_id="paper-1",
            base_trigger="manual",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=reserve,
            release_source=release,
        )

        self.assertEqual(await driver.run(), "fatal_stop")

    async def test_continuous_policy_can_idle_then_resume_exactly_once(self):
        calls = []
        wake_requested = False

        async def reserve(*_args):
            return "manager-owned-token"

        async def release(*_args):
            return None

        async def execute(index, trigger, prior_results, token):
            nonlocal wake_requested
            calls.append((index, trigger, prior_results, token))
            if index == 1:
                wake_requested = True
                return "continue_reset", None
            return "stopped", None

        driver = ProofRoundDriver(
            policy=ContinuousPruningPolicy(),
            source_type="paper",
            source_id="paper-1",
            base_trigger="manual",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=reserve,
            release_source=release,
        )

        self.assertEqual(await driver.run(), "stopped")
        self.assertTrue(wake_requested)
        self.assertEqual([call[0] for call in calls], [1, 2])
        self.assertEqual(calls[1][2], "")

    async def test_source_reset_clears_prior_summaries_without_stopping(self):
        calls = []

        async def reserve(*_args):
            return "owner-token"

        async def release(*_args):
            return None

        async def execute(index, _trigger, prior_results, _token):
            calls.append((index, prior_results))
            result = SimpleNamespace(
                verified_count=1,
                total_candidates=1,
                novel_count=1,
                results=[],
                had_error=False,
            )
            if index == 1:
                return "completed", result
            if index == 2:
                return "completed_reset", result
            self.assertEqual(prior_results.count("Round"), 1)
            return "stopped", None

        driver = ProofRoundDriver(
            policy=ContinuousPruningPolicy(),
            source_type="brainstorm",
            source_id="topic-1",
            base_trigger="manual",
            execute_round=execute,
            should_stop=lambda: False,
            reserve_source=reserve,
            release_source=release,
        )

        self.assertEqual(await driver.run(), "stopped")
        self.assertIn("Round 1:", calls[1][1])
        self.assertIn("Round 2:", calls[2][1])
        self.assertNotIn("Round 1:", calls[2][1])


if __name__ == "__main__":
    unittest.main()
