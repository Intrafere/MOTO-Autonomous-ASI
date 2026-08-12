import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from backend.autonomous.core.proof_run_manager import (
    ProofRunControl,
    ProofRunManager,
    ProofRunSourceInvalidError,
)
from backend.shared.models import ProofRunSnapshot


class ProofRunManagerTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.manager = ProofRunManager()
        self.control = ProofRunControl(
            snapshot=ProofRunSnapshot(
                proof_run_id="proof-run-1",
                run_mode="one_round",
                scope="manual",
                source_type="paper",
                source_id="paper-1",
                proof_store_id="manual:active",
                run_id="manual-run",
                lifecycle_generation=1,
                status="running",
            )
        )

    async def test_concurrent_updates_preserve_both_lifecycle_and_pruning(self):
        pruning_state = {
            "proof_run_id": "proof-run-1",
            "lifecycle_generation": 1,
            "status": "queued",
        }
        await asyncio.gather(
            self.manager.update(self.control, status="stopping"),
            self.manager.save_pruning_state(self.control, pruning_state),
        )
        self.assertEqual(self.control.snapshot.status, "stopping")
        self.assertEqual(self.control.snapshot.pruning_status, "queued")
        self.assertEqual(self.control.snapshot.pruning_state, pruning_state)

    async def test_stale_pruning_persistence_is_rejected(self):
        persisted = await self.manager.save_pruning_state(
            self.control,
            {
                "proof_run_id": "proof-run-1",
                "lifecycle_generation": 2,
                "status": "queued",
            },
        )
        self.assertFalse(persisted)
        self.assertIsNone(self.control.snapshot.pruning_state)

    async def test_candidate_checkpoint_is_process_local_and_source_scoped(self):
        checkpoint = {
            "source_type": "paper",
            "source_id": "paper-1",
            "status": "provider_paused",
            "candidates": [{"candidate": {"theorem_id": "t1"}}],
        }

        self.assertTrue(
            await self.manager.save_candidate_checkpoint(self.control, checkpoint)
        )
        self.assertEqual(
            await self.manager.load_candidate_checkpoint(self.control),
            checkpoint,
        )
        checkpoint["status"] = "mutated-after-save"
        self.assertEqual(
            (await self.manager.load_candidate_checkpoint(self.control))["status"],
            "provider_paused",
        )

        rejected = await self.manager.save_candidate_checkpoint(
            self.control,
            {
                "source_type": "paper",
                "source_id": "different-paper",
                "status": "provider_paused",
            },
        )
        self.assertFalse(rejected)
        self.assertEqual(
            (await self.manager.load_candidate_checkpoint(self.control))["source_id"],
            "paper-1",
        )

    async def test_continuous_round_metadata_is_bounded_in_memory(self):
        self.control.snapshot = self.control.snapshot.model_copy(
            update={"run_mode": "loop_with_pruning"}
        )
        await self.manager.complete_round(
            self.control,
            round_number=3,
            summary="s" * 5000,
            reference="r" * 700,
            candidate_checkpoint_reference="checkpoint-3.json",
            proof_set_revision=9,
        )
        snapshot = self.control.snapshot
        self.assertEqual(snapshot.current_round, 3)
        self.assertEqual(snapshot.last_completed_round, 3)
        self.assertEqual(len(snapshot.last_round_summary), 4000)
        self.assertEqual(len(snapshot.last_round_reference), 512)
        self.assertEqual(snapshot.proof_set_revision, 9)

    async def test_complete_round_clears_stale_candidate_checkpoint_at_round_boundary(self):
        await self.manager.save_candidate_checkpoint(
            self.control,
            {
                "source_type": "paper",
                "source_id": "paper-1",
                "status": "complete",
                "attempts_by_candidate": {"candidate-1": [{"attempt": 1}]},
            },
        )

        await self.manager.complete_round(
            self.control,
            round_number=2,
            candidate_checkpoint_reference="",
        )

        self.assertIsNone(await self.manager.load_candidate_checkpoint(self.control))

    async def test_complete_round_keeps_candidate_checkpoint_when_reference_remains(self):
        await self.manager.save_candidate_checkpoint(
            self.control,
            {
                "source_type": "paper",
                "source_id": "paper-1",
                "status": "provider_paused",
            },
        )

        await self.manager.complete_round(
            self.control,
            round_number=2,
            candidate_checkpoint_reference="proof-run-1:2:1",
        )

        checkpoint = await self.manager.load_candidate_checkpoint(self.control)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["status"], "provider_paused")

    async def test_round_completion_reports_candidates_and_continuation(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        self.control.snapshot = self.control.snapshot.model_copy(
            update={"run_mode": "loop_with_pruning"}
        )
        snapshot = await self.manager.complete_round(
            self.control,
            round_number=4,
            valid_candidate_count=0,
            summary="Round 4: 0/0 candidates verified, 0 novel.",
        )

        self.assertEqual(snapshot.last_completed_round, 4)
        event_type, payload = events[-1]
        self.assertEqual(event_type, "proof_run_round_complete")
        self.assertEqual(payload["candidate_count"], 0)
        self.assertTrue(payload["next_round_automatic"])
        self.assertIn("0/0 candidates", payload["round_summary"])

    async def test_round_completion_does_not_announce_continuation_after_fatal_reason(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        self.control.snapshot = self.control.snapshot.model_copy(
            update={
                "run_mode": "loop_with_pruning",
                "terminal_reason": "proof_output_truncation_recovery_exhausted",
            }
        )

        await self.manager.complete_round(
            self.control,
            round_number=2,
            valid_candidate_count=None,
        )

        self.assertFalse(events[-1][1]["next_round_automatic"])

    async def test_stop_wakes_wait_and_blocks_stale_nonterminal_update(self):
        self.control.snapshot = self.control.snapshot.model_copy(
            update={"run_mode": "loop_with_pruning", "status": "provider_paused"}
        )
        self.manager._runs["proof-run-1"] = self.control
        snapshot = await self.manager.stop("proof-run-1", 1)
        self.assertEqual(snapshot.status, "stopping")
        self.assertTrue(self.control.stop_event.is_set())
        self.assertTrue(self.control.wake_event.is_set())
        stale = await self.manager.update(
            self.control,
            expected_generation=1,
            status="running",
        )
        self.assertEqual(stale.status, "stopping")

    async def test_terminal_callback_is_exact_once(self):
        events = []

        async def callback(snapshot):
            events.append(snapshot.status)

        self.control.terminal_callback = callback
        await self.manager._finish(
            self.control,
            1,
            status="completed",
        )
        await self.manager._finish(
            self.control,
            1,
            status="completed",
        )
        self.assertEqual(events, ["completed"])
        self.assertTrue(self.control.snapshot.terminal_event_emitted)

    async def test_error_helper_preserves_terminal_reason(self):
        events = []

        async def callback(snapshot):
            events.append((snapshot.status, snapshot.terminal_reason))

        self.control.terminal_callback = callback
        snapshot = await self.manager.error(
            self.control,
            terminal_reason="proof_stage_error",
            reason="stage checkpoint preserved",
        )

        self.assertEqual(snapshot.status, "error")
        self.assertEqual(snapshot.terminal_reason, "proof_stage_error")
        self.assertEqual(snapshot.last_error_summary, "stage checkpoint preserved")
        self.assertEqual(events, [("error", "proof_stage_error")])

    async def test_continuous_lifecycle_events_are_typed_and_generation_scoped(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        self.control.snapshot = self.control.snapshot.model_copy(
            update={
                "run_mode": "loop_with_pruning",
                "round_limit": None,
                "unbounded": True,
            }
        )
        await self.manager.begin_round(self.control, round_number=2)
        await self.manager.complete_round(self.control, round_number=2)

        self.assertEqual(
            [event_type for event_type, _payload in events],
            [
                "proof_run_round_started",
                "proof_run_round_complete",
            ],
        )
        for _event_type, payload in events:
            self.assertEqual(payload["proof_run_id"], "proof-run-1")
            self.assertEqual(payload["run_mode"], "loop_with_pruning")
            self.assertEqual(payload["scope"], "manual")
            self.assertEqual(payload["source_type"], "paper")
            self.assertEqual(payload["source_id"], "paper-1")
            self.assertEqual(payload["run_id"], "manual-run")
            self.assertEqual(payload["round_index"], 2)
            self.assertIsNone(payload["round_limit"])
            self.assertTrue(payload["unbounded"])
            self.assertEqual(payload["lifecycle_generation"], 1)
        complete_payload = events[-1][1]
        self.assertIsNone(complete_payload["candidate_count"])
        self.assertTrue(complete_payload["next_round_automatic"])

    async def test_dedicated_terminal_reason_suppresses_generic_terminal_event(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        await self.manager._finish(
            self.control,
            1,
            status="error",
            terminal_reason="proof_output_truncation_recovery_exhausted",
        )
        self.assertEqual(events, [])

    async def test_source_invalid_has_typed_terminal_reason(self):
        self.control.cleaned_up = True

        async def worker(_control):
            raise ProofRunSourceInvalidError("Paper content not found")

        await self.manager._drive(self.control, worker)

        self.assertEqual(self.control.snapshot.status, "error")
        self.assertEqual(self.control.snapshot.terminal_reason, "source_invalid")
        self.assertIn("Paper content not found", self.control.snapshot.last_error_summary)

    async def test_hard_repair_is_terminal_and_not_resumable(self):
        snapshot = await self.manager.repair_required(
            self.control,
            reason="Repair the configured proof provider.",
        )
        self.assertEqual(snapshot.status, "error")
        self.assertEqual(snapshot.terminal_reason, "repair_required")
        self.assertIn("Repair the configured proof provider", snapshot.last_error_summary)

    async def test_shutdown_all_cancels_tasks_and_discards_pending_pruning(self):
        drain_values = []

        class Pruning:
            async def drain(self, *, preserve_pending):
                drain_values.append(preserve_pending)

        async def sleeping():
            await asyncio.sleep(60)

        self.control.pruning_coordinator = Pruning()
        self.control.task = asyncio.create_task(sleeping())
        self.control.reservation_token = "proof-run-1:owner"
        self.manager._runs["proof-run-1"] = self.control

        await self.manager.shutdown_all()

        self.assertTrue(self.control.task.cancelled())
        self.assertTrue(self.control.cleaned_up)
        self.assertEqual(drain_values, [False])

    async def test_legacy_lifecycle_state_is_purged_without_touching_other_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / "proof_runs"
            state_dir.mkdir()
            stale = state_dir / "proof-run-old.json"
            unrelated = state_dir / "session-checkpoint.json"
            proof = root / "manual_proofs" / "proof.json"
            proof.parent.mkdir()
            stale.write_text("{}", encoding="utf-8")
            unrelated.write_text("keep", encoding="utf-8")
            proof.write_text("keep", encoding="utf-8")

            await self.manager.purge_legacy_state(root)

            self.assertFalse(stale.exists())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
            self.assertEqual(proof.read_text(encoding="utf-8"), "keep")

    async def test_new_manager_does_not_load_prior_process_runs(self):
        self.manager._runs["proof-run-1"] = self.control
        self.assertIsNone(await ProofRunManager().get("proof-run-1"))

    async def test_collection_is_bounded_metadata_only_and_sorted(self):
        older = self.control.snapshot.model_copy(
            update={
                "updated_at": datetime(2026, 1, 1),
            }
        )
        newer = older.model_copy(
            update={
                "proof_run_id": "proof-run-2",
                "source_id": "paper-2",
                "updated_at": datetime(2026, 1, 2),
            }
        )
        self.manager._runs = {
            older.proof_run_id: ProofRunControl(snapshot=older),
            newer.proof_run_id: ProofRunControl(snapshot=newer),
        }

        result = await self.manager.list_runs(limit=1, scope="manual")

        self.assertEqual(result.count, 1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.runs[0].proof_run_id, "proof-run-2")
        self.assertNotIn(
            "proof_store_id",
            result.runs[0].model_dump(),
        )

    async def test_source_lookup_ignores_terminal_runs_for_ambiguity(self):
        continuous = self.control.snapshot.model_copy(
            update={
                "proof_run_id": "proof-run-loop",
                "run_mode": "loop_with_pruning",
                "status": "stopped",
                "updated_at": datetime(2026, 1, 2),
            }
        )
        active = self.control.snapshot.model_copy(
            update={"proof_run_id": "proof-run-active", "updated_at": datetime(2026, 1, 3)}
        )
        self.manager._runs = {
            continuous.proof_run_id: ProofRunControl(snapshot=continuous),
            active.proof_run_id: ProofRunControl(snapshot=active),
        }

        result = await self.manager.find_by_source(
            scope="manual",
            source_type="paper",
            source_id="paper-1",
            limit=20,
        )

        self.assertFalse(result.ambiguous)
        self.assertEqual(result.preferred_proof_run_id, "proof-run-active")

    async def test_source_lookup_prefers_active_run_outside_terminal_bound(self):
        newest_terminal = self.control.snapshot.model_copy(
            update={
                "proof_run_id": "proof-run-terminal-new",
                "status": "completed",
                "updated_at": datetime(2026, 1, 3),
            }
        )
        older_active = self.control.snapshot.model_copy(
            update={
                "proof_run_id": "proof-run-active",
                "run_mode": "loop_with_pruning",
                "status": "running",
                "updated_at": datetime(2026, 1, 2),
            }
        )
        self.manager._runs = {
            newest_terminal.proof_run_id: ProofRunControl(snapshot=newest_terminal),
            older_active.proof_run_id: ProofRunControl(snapshot=older_active),
        }

        result = await self.manager.find_by_source(
            scope="manual",
            source_type="paper",
            source_id="paper-1",
            limit=1,
        )

        self.assertTrue(result.truncated)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.preferred_proof_run_id, "proof-run-active")
        self.assertEqual(result.runs[0].proof_run_id, "proof-run-active")
