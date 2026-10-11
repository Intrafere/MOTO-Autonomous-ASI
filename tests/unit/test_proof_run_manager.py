import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from backend.autonomous.core.proof_run_manager import (
    NO_DISCOVERY_LIMIT_TERMINAL_REASON,
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

    async def test_continuous_no_discovery_streak_completes_on_tenth_round(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        self.control.snapshot = self.control.snapshot.model_copy(
            update={
                "run_mode": "loop_with_pruning",
                "no_discovery_round_limit": 10,
            }
        )

        for round_number in range(1, 10):
            snapshot = await self.manager.complete_round(
                self.control,
                round_number=round_number,
                valid_candidate_count=0,
            )
            self.assertEqual(snapshot.consecutive_no_discovery_rounds, round_number)
            self.assertFalse(snapshot.terminal_reason)
            self.assertTrue(events[-1][1]["next_round_automatic"])

        snapshot = await self.manager.complete_round(
            self.control,
            round_number=10,
            valid_candidate_count=0,
        )
        self.assertEqual(snapshot.consecutive_no_discovery_rounds, 10)
        self.assertEqual(
            snapshot.terminal_reason,
            NO_DISCOVERY_LIMIT_TERMINAL_REASON,
        )
        self.assertFalse(events[-1][1]["next_round_automatic"])

    async def test_continuous_discovery_resets_no_discovery_streak(self):
        self.control.snapshot = self.control.snapshot.model_copy(
            update={
                "run_mode": "loop_with_pruning",
                "no_discovery_round_limit": 10,
                "consecutive_no_discovery_rounds": 9,
            }
        )

        snapshot = await self.manager.complete_round(
            self.control,
            round_number=10,
            valid_candidate_count=1,
        )
        self.assertEqual(snapshot.consecutive_no_discovery_rounds, 0)
        self.assertFalse(snapshot.terminal_reason)

    async def test_failure_without_valid_candidate_count_does_not_advance_streak(self):
        self.control.snapshot = self.control.snapshot.model_copy(
            update={
                "run_mode": "loop_with_pruning",
                "no_discovery_round_limit": 10,
                "consecutive_no_discovery_rounds": 4,
            }
        )

        snapshot = await self.manager.complete_round(
            self.control,
            round_number=5,
            valid_candidate_count=None,
            summary="Provider failure",
        )
        self.assertEqual(snapshot.consecutive_no_discovery_rounds, 4)
        self.assertFalse(snapshot.terminal_reason)

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

    async def test_stop_cancels_in_flight_worker_without_waiting_for_model(self):
        worker_started = asyncio.Event()
        worker_cancelled = asyncio.Event()

        async def worker(_control):
            worker_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                worker_cancelled.set()
                raise

        self.control.cleaned_up = True
        self.manager._runs["proof-run-1"] = self.control
        self.control.task = asyncio.create_task(self.manager._drive(self.control, worker))
        await asyncio.wait_for(worker_started.wait(), timeout=1)

        snapshot = await self.manager.stop("proof-run-1", 1)

        self.assertEqual(snapshot.status, "stopping")
        await asyncio.wait_for(worker_cancelled.wait(), timeout=1)
        with self.assertRaises(asyncio.CancelledError):
            await self.control.task
        self.assertEqual(self.control.snapshot.status, "stopped")
        self.assertEqual(self.control.snapshot.terminal_reason, "cancelled")

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

    async def test_repair_terminal_event_and_reconnect_item_include_error_detail(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        self.manager._runs["proof-run-1"] = self.control
        snapshot = await self.manager.repair_required(
            self.control,
            reason=(
                "Prompt (51489 tokens) exceeds model's context window (5632 tokens). "
                "Increase LM Studio Context Length (n_ctx)."
            ),
        )

        self.assertEqual(snapshot.terminal_reason, "repair_required")
        self.assertEqual(events[-1][0], "proof_run_terminal")
        self.assertIn("5632", events[-1][1]["last_error_summary"])
        collection = await self.manager.find_by_source(
            scope="manual",
            source_type="paper",
            source_id="paper-1",
        )
        self.assertIn("n_ctx", collection.runs[0].last_error_summary)

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

    async def test_context_overflow_is_a_dedicated_terminal_error(self):
        events = []
        callbacks = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        async def terminal_callback(snapshot):
            callbacks.append((snapshot.status, snapshot.terminal_reason))

        self.control.event_callback = event_callback
        self.control.terminal_callback = terminal_callback
        snapshot = await self.manager.error(
            self.control,
            terminal_reason="context_overflow",
            reason="Mandatory proof context exceeded the configured input budget.",
        )

        self.assertEqual(snapshot.status, "error")
        self.assertEqual(snapshot.terminal_reason, "context_overflow")
        self.assertIn("Mandatory proof context", snapshot.last_error_summary)
        self.assertEqual(callbacks, [("error", "context_overflow")])
        self.assertEqual(events, [])

    async def test_source_invalid_has_typed_terminal_reason(self):
        self.control.cleaned_up = True

        async def worker(_control):
            raise ProofRunSourceInvalidError("Paper content not found")

        await self.manager._drive(self.control, worker)

        self.assertEqual(self.control.snapshot.status, "error")
        self.assertEqual(self.control.snapshot.terminal_reason, "source_invalid")
        self.assertIn("Paper content not found", self.control.snapshot.last_error_summary)

    async def test_provider_error_events_carry_route_identity_text(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        identity = "[Manual proof check · Proof Brainstorm · vendor/test-free:free via OpenRouter]"
        route_identity = {
            "workflow_mode": "manual_proof_check",
            "role_id": "manual_proof_brainstorm",
            "model": "vendor/test-free:free",
            "provider": "openrouter",
            "configured_host_provider": "",
            "host_routing": "auto",
            "route_identity": identity,
        }
        self.control.event_callback = event_callback
        await self.manager.provider_paused(self.control, route_identity=route_identity)
        snapshot = await self.manager.repair_required(
            self.control,
            reason="Repair the configured proof provider.",
            route_identity=route_identity,
        )

        paused = dict(events)["proof_run_provider_paused"]
        self.assertEqual(paused["message"], f"{identity} Proof run paused for provider credits.")
        self.assertEqual(paused["model"], "vendor/test-free:free")
        self.assertEqual(paused["configured_host_provider"], "")
        terminal = dict(events)["proof_run_terminal"]
        self.assertEqual(
            snapshot.last_error_summary,
            f"{identity} Repair the configured proof provider.",
        )
        self.assertEqual(terminal["last_error_summary"], snapshot.last_error_summary)
        self.assertEqual(terminal["route_identity"], identity)
        self.assertEqual(terminal["proof_run_id"], "proof-run-1")
        self.assertNotIn("route_identity", ProofRunSnapshot.model_fields)

    async def test_provider_pause_without_identity_keeps_legacy_payload(self):
        events = []

        async def event_callback(event_type, payload):
            events.append((event_type, payload))

        self.control.event_callback = event_callback
        await self.manager.provider_paused(self.control)

        paused = dict(events)["proof_run_provider_paused"]
        self.assertNotIn("message", paused)
        self.assertNotIn("route_identity", paused)

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

    async def test_published_activity_is_numbered_retained_and_replayable(self):
        sent = []

        async def send(event_type, data):
            sent.append((event_type, data))

        self.manager._runs = {self.control.snapshot.proof_run_id: self.control}
        broadcast = self.manager.activity_broadcaster(self.control, send)
        await broadcast("proof_candidate_list_review_started", {"list_attempt": 1})
        await broadcast("proof_candidate_list_review_rejected", {"list_attempt": 1})
        await broadcast("proof_candidate_list_regeneration_started", {"list_attempt": 2})

        self.assertEqual(
            [data["proof_run_activity_sequence"] for _, data in sent],
            [1, 2, 3],
        )
        self.assertTrue(all(data["proof_run_id"] == "proof-run-1" for _, data in sent))

        replay = await self.manager.get_activity("proof-run-1", after_sequence=1)
        self.assertEqual(replay.latest_sequence, 3)
        self.assertEqual([event.sequence for event in replay.events], [2, 3])
        self.assertEqual(
            [event.type for event in replay.events],
            [
                "proof_candidate_list_review_rejected",
                "proof_candidate_list_regeneration_started",
            ],
        )
        self.assertEqual(replay.events[1].data["list_attempt"], 2)
        self.assertFalse(replay.gap)
        self.assertFalse(replay.truncated)

        baseline = await self.manager.get_activity("proof-run-1", limit=0)
        self.assertEqual(baseline.events, [])
        self.assertTrue(baseline.truncated)
        self.assertEqual(baseline.latest_sequence, 3)

    async def test_activity_replay_reports_evicted_events_as_a_gap(self):
        async def send(event_type, data):
            return None

        self.manager._runs = {self.control.snapshot.proof_run_id: self.control}
        for attempt in range(1, 306):
            await self.manager.publish_activity(
                self.control,
                send,
                "proof_candidate_list_review_started",
                {"list_attempt": attempt},
            )

        replay = await self.manager.get_activity("proof-run-1", after_sequence=2)
        self.assertTrue(replay.gap)
        self.assertEqual(replay.oldest_retained_sequence, 6)
        self.assertEqual(replay.events[0].sequence, 6)
        self.assertTrue(replay.truncated)

    async def test_disabled_journal_publishes_without_sequence(self):
        sent = []

        async def send(event_type, data):
            sent.append(data)

        self.control.activity_journal_enabled = False
        self.manager._runs = {self.control.snapshot.proof_run_id: self.control}
        await self.manager.publish_activity(self.control, send, "proof_check_started", {})

        self.assertNotIn("proof_run_activity_sequence", sent[0])
        replay = await self.manager.get_activity("proof-run-1")
        self.assertEqual(replay.latest_sequence, 0)
        self.assertEqual(replay.events, [])

    async def test_lifecycle_events_join_the_activity_journal(self):
        sent = []

        async def send(event_type, data):
            sent.append(event_type)

        self.control.event_callback = send
        self.manager._runs = {self.control.snapshot.proof_run_id: self.control}
        await self.manager.begin_round(self.control, round_number=2)

        replay = await self.manager.get_activity("proof-run-1")
        self.assertEqual(sent, ["proof_run_round_started"])
        self.assertEqual([event.type for event in replay.events], ["proof_run_round_started"])
        self.assertEqual(replay.events[0].data["round_index"], 2)

    async def test_only_recent_finished_runs_keep_activity_journals(self):
        async def send(event_type, data):
            return None

        controls = []
        for index in range(7):
            control = ProofRunControl(
                snapshot=self.control.snapshot.model_copy(
                    update={
                        "proof_run_id": f"proof-run-{index}",
                        "status": "completed",
                        "updated_at": datetime(2026, 1, 1 + index),
                    }
                ),
                cleaned_up=True,
            )
            await self.manager.publish_activity(control, send, "proof_check_started", {})
            controls.append(control)
        self.manager._runs = {
            control.snapshot.proof_run_id: control for control in controls
        }

        await self.manager._trim_terminal_activity_journals()

        self.assertEqual(
            [bool(control.activity_journal) for control in controls],
            [False, False, True, True, True, True, True],
        )
        cleared = await self.manager.get_activity("proof-run-0")
        self.assertTrue(cleared.gap)
        self.assertEqual(cleared.latest_sequence, 1)

    async def test_unknown_run_activity_is_none(self):
        self.assertIsNone(await self.manager.get_activity("proof-run-missing"))
