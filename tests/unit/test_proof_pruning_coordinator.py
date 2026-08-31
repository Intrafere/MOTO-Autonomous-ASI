import asyncio
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase

from backend.autonomous.core.proof_pruning_coordinator import (
    ProofPruningCoordinator,
)
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import (
    ProofPruneContextPressure,
    ProofPruneProposal,
    ProofPruneReviewResult,
    ProofRecord,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
)


def runtime_snapshot() -> ProofRuntimeConfigSnapshot:
    role = ProofRoleConfigSnapshot(
        provider="lm_studio",
        model_id="test-model",
        context_window=8192,
        max_output_tokens=1024,
    )
    return ProofRuntimeConfigSnapshot(
        brainstorm=role,
        paper=role,
        validator=role,
        assistant=role,
    )


class HeldReviewService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def review(self, snapshot, *, event_callback=None):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return ProofPruneReviewResult(
            outcome="no_prune",
            proposal=ProofPruneProposal(
                action="no_prune",
                proof_id=None,
                expected_theorem_hash=None,
                expected_lean_hash=None,
                reasoning="Every active proof remains useful.",
            ),
        )


class FailingReviewService:
    async def review(self, snapshot, *, event_callback=None):
        raise RuntimeError("reproduced pruning failure")


class SecretFailingReviewService:
    async def review(self, snapshot, *, event_callback=None):
        raise RuntimeError("Bearer sk-secret-value")


class ProofPruningCoordinatorTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = ProofDatabase()
        self.database.set_base_dir(Path(self.tempdir.name))
        await self.database.initialize()
        self.events = []
        self.persisted = None
        self.service = HeldReviewService()
        self.coordinator = ProofPruningCoordinator(
            proof_database=self.database,
            runtime_snapshot=runtime_snapshot(),
            proof_run_id="proof-run-1",
            run_mode="one_round",
            run_id="owning-run",
            lifecycle_generation=1,
            scope="manual",
            source_type="paper",
            source_id="paper-one",
            canonical_user_prompt="Prove the objective.",
            proof_store_id="manual:active",
            broadcast_fn=self._broadcast,
            persist_fn=self._persist,
            review_service=self.service,
        )

    async def asyncTearDown(self) -> None:
        await self.coordinator.drain(preserve_pending=False)
        self.tempdir.cleanup()

    async def _broadcast(self, event, payload):
        self.events.append((event, payload))

    async def _persist(self, payload):
        self.persisted = payload
        return True

    @staticmethod
    def proof(proof_id: str, novelty_tier: str = "mathematical_discovery"):
        return ProofRecord(
            proof_id=proof_id,
            theorem_statement="True",
            theorem_name=proof_id,
            source_type="paper",
            source_id="paper-one",
            run_id="owning-run",
            lean_code=f"theorem {proof_id.replace('-', '_')} : True := by trivial",
            novel=novelty_tier != "not_novel",
            novelty_tier=novelty_tier,
        )

    async def test_third_eligible_registration_schedules_without_blocking(self):
        await self.coordinator.on_proof_registered(self.proof("p1"))
        await self.coordinator.on_proof_registered(self.proof("p2"))
        self.assertIsNone(self.coordinator.active_task)

        await asyncio.wait_for(
            self.coordinator.on_proof_registered(self.proof("p3")),
            timeout=0.2,
        )
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        self.assertEqual(self.service.calls, 1)
        self.assertFalse(self.coordinator.active_task.done())
        revision_before = await self.database.get_proof_set_revision()
        proofs_before = await self.database.get_all_proofs()

        self.service.release.set()
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)
        self.assertEqual(
            self.coordinator.state.last_scheduled_acceptance_baseline,
            3,
        )
        self.assertIn(
            "proof_prune_no_change",
            [event for event, _payload in self.events],
        )
        self.assertEqual(
            await self.database.get_proof_set_revision(),
            revision_before,
        )
        self.assertEqual(await self.database.get_all_proofs(), proofs_before)

    async def test_known_duplicate_and_replayed_ids_do_not_count(self):
        await self.coordinator.on_proof_registered(
            self.proof("known", "not_novel")
        )
        await self.coordinator.on_proof_registered(
            self.proof("duplicate", "duplicate_novel")
        )
        await self.coordinator.on_proof_registered(self.proof("p1"))
        await self.coordinator.on_proof_registered(self.proof("p1"))
        self.assertEqual(self.coordinator.state.accepted_prompt_novel_total, 1)
        self.assertIsNone(self.coordinator.active_task)

    async def test_failure_event_exposes_safe_diagnostic(self):
        self.coordinator.review_service = FailingReviewService()
        pressure = ProofPruneContextPressure(
            trigger="context_pressure",
            prompt_tokens=9000,
            available_input_tokens=8000,
            active_proof_tokens=1200,
            active_proof_context_tokens=1200,
            configured_context_window=9000,
            proof_set_revision=0,
        )
        await self.coordinator.on_context_pressure(
            pressure,
            urgent=True,
            proof_set_revision=0,
        )
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)
        event, payload = next(
            item for item in self.events if item[0] == "proof_prune_error"
        )
        self.assertEqual(event, "proof_prune_error")
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertEqual(payload["error_summary"], "reproduced pruning failure")
        self.assertIn("RuntimeError: reproduced pruning failure", payload["message"])

    async def test_failed_cadence_review_restores_threshold(self):
        self.coordinator.review_service = FailingReviewService()
        for proof_id in ("p1", "p2", "p3"):
            await self.coordinator.on_proof_registered(self.proof(proof_id))
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)

        self.assertEqual(
            self.coordinator.state.last_scheduled_acceptance_baseline,
            0,
        )
        self.assertTrue(self.coordinator.state.follow_up_required)

    async def test_failure_event_redacts_secret_like_exception_text(self):
        self.coordinator.review_service = SecretFailingReviewService()
        pressure = ProofPruneContextPressure(
            trigger="context_pressure",
            active_proof_tokens=1200,
            active_proof_context_tokens=1200,
        )
        await self.coordinator.on_context_pressure(
            pressure,
            urgent=True,
            proof_set_revision=0,
        )
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)
        _event, payload = next(
            item for item in self.events if item[0] == "proof_prune_error"
        )
        self.assertNotIn("sk-secret-value", payload["error_summary"])

    async def test_triggers_during_active_review_coalesce_one_follow_up(self):
        for proof_id in ("p1", "p2", "p3"):
            await self.coordinator.on_proof_registered(self.proof(proof_id))
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        for proof_id in ("p4", "p5", "p6"):
            await self.coordinator.on_proof_registered(self.proof(proof_id))
        self.assertTrue(self.coordinator.state.follow_up_required)

        self.service.release.set()
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)
        self.assertEqual(self.service.calls, 2)
        self.assertEqual(
            self.coordinator.state.last_scheduled_acceptance_baseline,
            6,
        )

    async def test_drain_cancels_held_review_and_prevents_late_work(self):
        for proof_id in ("p1", "p2", "p3"):
            await self.coordinator.on_proof_registered(self.proof(proof_id))
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        self.assertEqual(
            self.coordinator.state.active_trigger_reasons,
            ["three_novel_proofs"],
        )
        await asyncio.wait_for(
            self.coordinator.drain(preserve_pending=True),
            timeout=1,
        )
        self.assertTrue(self.coordinator.active_task.done())
        self.assertEqual(self.coordinator.state.status, "queued")
        self.assertEqual(
            self.coordinator.state.queued_trigger_reasons,
            ["three_novel_proofs"],
        )
        self.assertEqual(self.coordinator.state.active_trigger_reasons, [])
        self.assertEqual(
            self.persisted["queued_trigger_reasons"],
            ["three_novel_proofs"],
        )

    async def test_drain_requeues_active_trigger_for_next_round_rebind(self):
        for proof_id in ("p1", "p2", "p3"):
            await self.coordinator.on_proof_registered(self.proof(proof_id))
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        await self.coordinator.drain(preserve_pending=True)
        payload = dict(self.persisted)

        rebound_service = HeldReviewService()
        rebound = ProofPruningCoordinator(
            proof_database=self.database,
            runtime_snapshot=runtime_snapshot(),
            proof_run_id="proof-run-1",
            run_mode="one_round",
            run_id="owning-run",
            lifecycle_generation=2,
            scope="manual",
            source_type="paper",
            source_id="paper-one",
            canonical_user_prompt="Prove the objective.",
            proof_store_id="manual:active",
            round_index=2,
            load_fn=lambda: asyncio.sleep(0, result=payload),
            persist_fn=self._persist,
            review_service=rebound_service,
        )
        await rebound.restore()
        await asyncio.wait_for(rebound_service.started.wait(), timeout=1)
        self.assertEqual(rebound.state.round_index, 2)
        self.assertEqual(
            rebound.state.active_trigger_reasons,
            ["three_novel_proofs"],
        )
        await rebound.drain(preserve_pending=False)

    async def test_unchanged_pressure_same_revision_is_not_rescheduled(self):
        pressure = ProofPruneContextPressure(
            trigger="context_pressure",
            prompt_tokens=7900,
            available_input_tokens=8000,
            active_proof_tokens=1200,
            active_proof_context_tokens=1200,
            configured_context_window=9000,
            route_config_fingerprint="route-a",
            proof_set_revision=0,
        )
        await self.coordinator.on_context_pressure(
            pressure,
            proof_set_revision=0,
        )
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        self.service.release.set()
        await asyncio.wait_for(self.coordinator.active_task, timeout=1)
        calls_after_first = self.service.calls

        await self.coordinator.on_context_pressure(
            pressure.model_copy(deep=True),
            proof_set_revision=0,
        )
        await asyncio.sleep(0)
        self.assertEqual(self.service.calls, calls_after_first)

    async def test_notify_registration_is_immediate_and_owned(self):
        for proof_id in ("p1", "p2"):
            await self.coordinator.notify_proof_registered(self.proof(proof_id))
        await asyncio.wait_for(
            self.coordinator.notify_proof_registered(self.proof("p3")),
            timeout=0.05,
        )
        await asyncio.wait_for(self.service.started.wait(), timeout=1)
        self.assertFalse(self.coordinator.active_task.done())

    async def test_restore_rebinds_compatible_state_to_new_lifecycle(self):
        payload = self.coordinator.state.model_copy(
            update={
                "lifecycle_generation": 1,
                "status": "queued",
                "queued_trigger_reasons": ["three_novel_proofs"],
            }
        ).model_dump(mode="json")
        restored = ProofPruningCoordinator(
            proof_database=self.database,
            runtime_snapshot=runtime_snapshot(),
            proof_run_id="proof-run-1",
            run_mode="one_round",
            run_id="owning-run",
            lifecycle_generation=2,
            scope="manual",
            source_type="paper",
            source_id="paper-one",
            canonical_user_prompt="Prove the objective.",
            proof_store_id="manual:active",
            load_fn=lambda: asyncio.sleep(0, result=payload),
            persist_fn=self._persist,
            review_service=self.service,
        )
        await restored.restore()
        self.assertEqual(restored.state.lifecycle_generation, 2)
        await restored.drain(preserve_pending=False)

    async def test_restore_rejects_source_identity_mismatch(self):
        payload = self.coordinator.state.model_copy(
            update={"source_id": "different-paper"}
        ).model_dump(mode="json")
        restored = ProofPruningCoordinator(
            proof_database=self.database,
            runtime_snapshot=runtime_snapshot(),
            proof_run_id="proof-run-1",
            run_mode="one_round",
            run_id="owning-run",
            lifecycle_generation=1,
            scope="manual",
            source_type="paper",
            source_id="paper-one",
            canonical_user_prompt="Prove the objective.",
            proof_store_id="manual:active",
            load_fn=lambda: asyncio.sleep(0, result=payload),
            persist_fn=self._persist,
            review_service=self.service,
        )
        await restored.restore()
        self.assertEqual(restored.state.source_id, "paper-one")
